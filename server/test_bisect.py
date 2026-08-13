"""Server Test #2: bisection over cap at fixed iteration budget.

Trains cap=8..12 (or an arbitrary --caps range) for --iters iterations each and
reports n_infos vs exploitability, so we can see in log-log whether expl grows
smoothly (pure scaling) or jumps at some cap (structural threshold, e.g. a
TURN_ACTIONS/stall edge case).

Runs caps in parallel via fork (Linux). ~20-30 min total on 32 vCPU.
Each worker must stay on ONE core: pin BLAS thread counts to 1 before numpy.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from cote_megaverse.solver1v1 import Solver
from cote_megaverse.rules import Type


def worker(job):
    cap, iters, gamma = job
    s = Solver(Type.C, Type.C, 2000, 2000, 10000, 10000, cap=cap,
               stall_cap=3, dtype="float32", gamma=gamma)
    t0 = time.time()
    for _ in range(iters):
        s.iteration()
    eA, eB = s.true_exploitability()
    return {"cap": cap, "n_states": len(s.states), "n_infos": s.n_infos,
            "explA": round(eA, 4), "explB": round(eB, 4),
            "total": round(eA + eB, 4), "secs": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default="8,9,10,11,12")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--jobs", type=int, default=5)
    a = ap.parse_args()
    caps = [int(c) for c in a.caps.split(",")]
    jobs = [(cap, a.iters, a.gamma) for cap in caps]
    t0 = time.time()
    with Pool(a.jobs) as pool:
        rows = pool.map(worker, jobs)
    print("cap  states     infos     explA   explB    total    secs", flush=True)
    for r in sorted(rows, key=lambda x: x["cap"]):
        print("%3d  %7d  %7d  %.4f  %.4f  %.4f  %6.1f"
              % (r["cap"], r["n_states"], r["n_infos"], r["explA"],
                 r["explB"], r["total"], r["secs"]), flush=True)
    print("wall=%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()