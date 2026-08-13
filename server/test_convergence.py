"""Server Test #1/#3: long-run convergence with exploitability trajectory.

Continues the CFR from a checkpoint if one exists (warm continue), otherwise
starts fresh. Every ``--report-every`` iterations it prints the gamma-consistent
true_exploitability and saves a checkpoint, so a long run can be resumed.

Usage (Linux, fork-based):
    python server/test_convergence.py --cap 12 --iters 5000 --report-every 100
    python server/test_convergence.py --cap 16 --iters 8450 --report-every 100

The cap=12 run (~97 min on 1 core) answers "more iterations or structural bug?"
The cap=16 run (~4.5 h) checks the "iterations per infoset" density hypothesis
(0.1017 * 83103 ~= 8450).
Single worker, but pin BLAS threads to 1 so the shared core is fully ours.
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

from cote_megaverse.solver1v1 import Solver
from cote_megaverse.rules import Type


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
    ap.add_argument("--cap", type=int, default=12)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--report-every", type=int, default=100)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--out", default="server_out")
    ap.add_argument("--hp", type=int, default=10000)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    params = {"typeA": Type.C, "typeB": Type.C, "atkA": 2000, "atkB": 2000,
              "hpA": a.hp, "hpB": a.hp, "cap": a.cap, "stall_cap": 3,
              "cfr_plus": True, "dtype": "float32", "gamma": a.gamma}
    # params persist with the checkpoint for resume (kept in a sidecar file)
    with open(os.path.join(a.out, "params_cap%d.pkl" % a.cap), "wb") as fh:
        pickle.dump(params, fh)

    ckpt = os.path.join(a.out, "ckpt_cap%d.pkl" % a.cap)
    s = load_checkpoint(ckpt, params)
    start = s._t if s else 0
    if s is None:
        s = Solver(**params)
        print("fresh solver cap=%d states=%d infos=%d"
              % (a.cap, len(s.states), s.n_infos), flush=True)
    else:
        print("resumed at _t=%d cap=%d" % (s._t, a.cap), flush=True)

    t0 = time.time()
    rows = []
    if start > 0:
        eA, eB = s.true_exploitability()
        rows.append((start, eA, eB, 0.0))
        print("iter=%5d  explA=%.4f explB=%.4f total=%.4f (%.0fs)  [resume]"
              % (start, eA, eB, eA + eB, 0.0), flush=True)
    for it in range(start + 1, a.iters + 1):
        s.iteration()
        if it % a.report_every == 0:
            t1 = time.time()
            eA, eB = s.true_exploitability()
            rows.append((it, eA, eB, time.time() - t1))
            print("iter=%5d  explA=%.4f explB=%.4f total=%.4f  (expl took %.1fs, run %.1fs)"
                  % (it, eA, eB, eA + eB, time.time() - t1, time.time() - t0), flush=True)
            save_checkpoint(s, ckpt)

    with open(os.path.join(a.out, "traj_cap%d.csv" % a.cap), "w") as fh:
        fh.write("iter,explA,explB,total,expl_time\n")
        for it, eA, eB, et in rows:
            fh.write("%d,%.6f,%.6f,%.6f,%.1f\n" % (it, eA, eB, eA + eB, et))
    print("done: %s" % os.path.join(a.out, "traj_cap%d.csv" % a.cap), flush=True)


if __name__ == "__main__":
    main()