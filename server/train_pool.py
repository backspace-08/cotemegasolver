"""Server: train the 27-solver pool (9 hits pairs x 3 turn modes).

Wraps SolverPool; all 27 solvers train in parallel on the 32 vCPU server.
Run this AFTER Test #1/#2 confirm the convergence story, since the pool needs
enough iterations to be near-Nash.

    python server/train_pool.py --out artifacts/solver_pool --iters 5000 \
        --cap 16 --gamma 0.995 --jobs 30

Notes: OMP_NUM_THREADS=1 is set inside solver_pool.py before numpy import so
workers do not oversubscribe the memory bus. Results -> out/pool.pkl.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from cote_megaverse.solver_pool import SolverPool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/solver_pool")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--cap", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--jobs", type=int, default=30)
    a = ap.parse_args()
    pool = SolverPool(out_dir=a.out, cap=a.cap, gamma=a.gamma, n_jobs=a.jobs)
    print("pool: %d solvers (cap=%d, gamma=%.3f) x %d iters on %d workers"
          % (len(pool.keys), a.cap, a.gamma, a.iters, a.jobs), flush=True)
    pool.train(iters=a.iters)


if __name__ == "__main__":
    main()