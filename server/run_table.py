"""Parallel 1v1 table runner (full-history solver, one job per worker).

Farms a grid of 1v1 matchups across ``--workers`` processes. Each job builds a
full-history tree solver (see src/cote_megaverse/solver_tree_fh.py), runs
``--iters`` iterations, and records the equilibrium value (A's perspective),
exploitability, outcome rates and the opening strategy. Optionally dumps the
per-state equilibrium value table (the continuation values the 2v2/3v3 layer
would look up).

Use the server's cores to run many independent duels in parallel — a single
run is memory-bound (threads do not help), but jobs are independent.

Memory: a cap=6 hp=10000 full-history solve holds ~2-4 GB (5.1M nodes).
Keep workers * ram <= server RAM.

Usage (Linux, from the repo root):
    # all type/atk/hp/first-move combos from the pools, sampled to 48 jobs:
    python server/run_table.py --workers 30 --max-jobs 48 \
        --type-pool A,B,C,D --atk-pool 1900,2000,2100 --hp-pool 5700,6000,6300 \
        --first-move 0,1 --iters 400 --out table_out

    # single matchup + value table dump:
    python server/run_table.py --type-pool C --atk-pool 2000 --hp-pool 10000 \
        --first-move 0,1 --iters 500 --dump-value --out table_out

Output in ``--out``:
    summary.csv            one row per job
    job_<id>.json          per-job detail (value, expl, opening)
    value_table_cap<cap>_fm<first>.csv   per-state (hA..to_move, value) dumps
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from cote_megaverse.rules import Type


def value_table(s, path):
    """Per-state equilibrium values of the average profile, vectorized."""
    n = s.n_states
    v = np.array(s.term_arr, dtype=np.float64)
    tot = s.avg.sum(axis=1, keepdims=True)
    nv = np.maximum(s.n_acts_info, 1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        avgn = np.where(tot > 0, s.avg / tot, 1.0 / nv)
    for data in reversed(s._value_data):
        if data is None:
            continue
        nt, cp, mask = data
        vc = np.where(mask, v[np.where(mask, cp, 0)], 0.0)
        sa = avgn[s.info_id[nt]]
        v[nt] = s.gamma * (sa * vc).sum(axis=1)
    with open(path, "w") as fh:
        fh.write("hA,hB,bankA,bankB,shA,shB,turn,to_move,value\n")
        for i in range(n):
            if s.term[i] is not None:
                continue
            st = s.states[i]
            fh.write("%d,%d,%d,%d,%d,%d,%d,%d,%.6f\n"
                     % (st[0], st[1], st[2], st[3], st[4], st[5], st[6], st[7],
                        v[i]))


def solve_one(job):
    from cote_megaverse.solver_tree_fh import Solver
    cap = job["cap"]
    s = Solver(Type(job["typeA"]), Type(job["typeB"]),
               job["atkA"], job["atkB"], job["hpA"], job["hpB"],
               cap=cap, stall_cap=3, first_move=job["first_move"],
               cfr_plus=True, dtype="float32", gamma=job["gamma"],
               full_history=True)
    t0 = time.time()
    for _ in range(job["iters"]):
        s.iteration()
    eA, eB = s.true_exploitability()
    val = s.profile_value()
    w, l, d = s.outcome_rates()
    opening = [(list(a), round(p, 4)) for a, p in s.start_strategy()
               if p > 0.01]
    res = dict(job)
    res.update({
        "hitsA": int(s.hA0), "hitsB": int(s.hB0),
        "value": round(val, 5),
        "explA": round(eA, 5), "explB": round(eB, 5),
        "total_expl": round(eA + eB, 5),
        "winA": round(w, 4), "winB": round(l, 4), "draw": round(d, 4),
        "nodes": len(s.states), "infos": int(s.n_infos),
        "secs": round(time.time() - t0, 1),
        "opening": opening,
    })
    if job.get("dump_value"):
        value_table(s, job["value_path"])
    return res


def build_jobs(a):
    jobs = []
    tA = [int(getattr(Type, x)) for x in a.type_pool.split(",")]
    tB = [int(getattr(Type, x)) for x in a.type_pool.split(",")]
    atk = [int(x) for x in a.atk_pool.split(",")]
    hp = [int(x) for x in a.hp_pool.split(",")]
    fm = [int(x) for x in a.first_move.split(",")]
    for ta in tA:
        for tb in tB:
            for aa in atk:
                for ab in atk:
                    for ha in hp:
                        for hb in hp:
                            for f in fm:
                                jobs.append({
                                    "typeA": ta, "typeB": tb,
                                    "atkA": aa, "atkB": ab,
                                    "hpA": ha, "hpB": hb,
                                    "first_move": f,
                                    "cap": a.cap, "iters": a.iters,
                                    "gamma": a.gamma,
                                })
    if a.max_jobs and len(jobs) > a.max_jobs:
        import random
        rng = random.Random(a.seed)
        jobs = rng.sample(jobs, a.max_jobs)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=6)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--type-pool", default="C")
    ap.add_argument("--atk-pool", default="2000")
    ap.add_argument("--hp-pool", default="10000")
    ap.add_argument("--first-move", default="0,1")
    ap.add_argument("--max-jobs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dump-value", action="store_true")
    ap.add_argument("--out", default="table_out")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    jobs = build_jobs(a)
    for i, j in enumerate(jobs):
        j["id"] = i
        if a.dump_value:
            j["dump_value"] = True
            j["value_path"] = os.path.join(
                a.out, "value_table_cap%d_fm%d.json" % (a.cap, j["first_move"]))

    print("jobs=%d workers=%d cap=%d iters=%d" % (len(jobs), a.workers, a.cap,
                                                  a.iters), flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        results = pool.map(solve_one, jobs)

    with open(os.path.join(a.out, "summary.csv"), "w") as fh:
        fh.write("id,typeA,typeB,atkA,atkB,hpA,hpB,first_move,hitsA,hitsB,"
                 "value,explA,explB,total_expl,winA,winB,draw,nodes,infos,secs\n")
        for r in sorted(results, key=lambda r: r["id"]):
            fh.write("%d,%s,%s,%d,%d,%d,%d,%d,%d,%d,%.5f,%.5f,%.5f,%.5f,"
                     "%.4f,%.4f,%.4f,%d,%d,%.1f\n"
                     % (r["id"], Type(r["typeA"]).name, Type(r["typeB"]).name,
                        r["atkA"], r["atkB"], r["hpA"], r["hpB"],
                        r["first_move"], r["hitsA"], r["hitsB"], r["value"],
                        r["explA"], r["explB"], r["total_expl"], r["winA"],
                        r["winB"], r["draw"], r["nodes"], r["infos"], r["secs"]))
            with open(os.path.join(a.out, "job_%d.json" % r["id"]), "w") as jf:
                json.dump(r, jf, indent=1)
    print("done in %.1fs -> %s/summary.csv" % (time.time() - t0, a.out),
          flush=True)


if __name__ == "__main__":
    main()
