"""Server run: full-history (perfect-recall) 1v1 tree solver with checkpoints.

Solves the 1v1 game on the FULL game tree with perfect-recall info sets
(see src/cote_megaverse/solver_tree_fh.py). Reports the gamma-consistent
exploitability every ``--report-every`` iterations and saves a checkpoint so a
long run can be resumed. The checkpoint (avg/regret) is the equilibrium
approximation the bot would play from.

Usage (Linux, run from the repo root; BLAS threads are pinned to 1):
    python server/run_full_history.py --cap 6 --hp 10000 --iters 600 --report-every 100
    python server/run_full_history.py --cap 6 --iters 400 --resume

Measured (cap=6 hp=10000, full history): build ~10s, ~0.4 it/s on 1 core,
expl 0.67@50 -> 0.09@100 -> ~0.05@300. Practical target ~0.01 needs ~500-600
iterations (~20-25 min on 1 core). Iteration is memory-bound; more threads do
not help a single run.
"""
import argparse
import os
import pickle
import sys
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from cote_megaverse.rules import Type
from cote_megaverse.solver_tree_fh import Solver


def load_checkpoint(path, params):
    if not os.path.exists(path):
        return None
    data = pickle.load(open(path, "rb"))
    s = Solver(**params)
    s.avg = data["avg"]
    s.regret = data["regret"]
    s._t = data["_t"]
    return s


def save_checkpoint(s, path):
    pickle.dump({"avg": s.avg, "regret": s.regret, "_t": s._t},
                open(path, "wb"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=6)
    ap.add_argument("--hp", type=int, default=10000)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--report-every", type=int, default=100)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--out", default="server_out_fh")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    params = {"typeA": Type.C, "typeB": Type.C, "atkA": 2000, "atkB": 2000,
              "hpA": a.hp, "hpB": a.hp, "cap": a.cap, "stall_cap": 3,
              "cfr_plus": True, "dtype": "float32", "gamma": a.gamma,
              "full_history": True}
    with open(os.path.join(a.out, "params_fh_cap%d.pkl" % a.cap), "wb") as fh:
        pickle.dump(params, fh)

    ckpt = os.path.join(a.out, "ckpt_fh_cap%d.pkl" % a.cap)
    if a.resume:
        s = load_checkpoint(ckpt, params)
        if s is None:
            print("no checkpoint found at %s; starting fresh" % ckpt, flush=True)
            s = Solver(**params)
        else:
            print("resumed at _t=%d cap=%d" % (s._t, a.cap), flush=True)
    else:
        s = Solver(**params)
    print("fresh solver cap=%d hp=%d full_history: nodes=%d infos=%d"
          % (a.cap, a.hp, len(s.states), s.n_infos), flush=True)

    start = s._t
    t0 = time.time()
    rows = []
    if start > 0:
        eA, eB = s.true_exploitability()
        rows.append((start, eA, eB, 0.0))
        print("iter=%5d  explA=%.4f explB=%.4f total=%.4f  [resume]"
              % (start, eA, eB, eA + eB), flush=True)
    for it in range(start + 1, a.iters + 1):
        s.iteration()
        if it % a.report_every == 0:
            t1 = time.time()
            eA, eB = s.true_exploitability()
            rows.append((it, eA, eB, time.time() - t1))
            print("iter=%5d  explA=%.4f explB=%.4f total=%.4f  (expl %.1fs, "
                  "run %.1fs)" % (it, eA, eB, eA + eB, time.time() - t1,
                                  time.time() - t0), flush=True)
            save_checkpoint(s, ckpt)

    with open(os.path.join(a.out, "traj_fh_cap%d.csv" % a.cap), "w") as fh:
        fh.write("iter,explA,explB,total,expl_time\n")
        for it, eA, eB, et in rows:
            fh.write("%d,%.6f,%.6f,%.6f,%.1f\n" % (it, eA, eB, eA + eB, et))

    eA, eB = s.true_exploitability()
    w, l, d = s.outcome_rates()
    print("\nFINAL iter=%d  explA=%.4f explB=%.4f total=%.4f"
          % (a.iters, eA, eB, eA + eB), flush=True)
    print("avg-profile W=%.3f L=%.3f D=%.3f" % (w, l, d), flush=True)
    print("checkpoint (equilibrium avg/regret): %s" % ckpt, flush=True)
    print("trajectory: %s" % os.path.join(a.out, "traj_fh_cap%d.csv" % a.cap),
          flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
