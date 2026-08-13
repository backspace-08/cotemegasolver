"""SolverPool: train and serve the hits-abstraction 1v1 solver pool.

Pool = 9 full-HP hits pairs (hA0,hB0 in 3..5) x turn regimes. Each solver is an
independent CFR game; workers train one key each in separate processes. Memory
rules enforced here:

- numpy CFR matrices use np.float32 (half of regret/avg/sig/_cfv_sum).
- OMP_NUM_THREADS is pinned to 1 before numpy import so parallel workers never
  oversubscribe BLAS threads on the memory bus.
- After the last iteration a worker calls ``release_graph()``, dropping the
  Python graph structures (~0.3 GB) so a heavy cap=20 solver trains in ~0.3 GB
  instead of ~0.9 GB; only avg/regret/_t are returned to the parent.
- Keys, not graphs, travel between parent and workers.

Results are pickled to out_dir/pool.pkl as {key: {params, avg, regret, _t}}.
"""
import os
import pickle
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cote_megaverse.solver1v1 import Solver  # noqa: E402
from cote_megaverse.rules import Type  # noqa: E402

# Budget regimes of the real game (rules.TURN_ACTIONS):
#   turn 1 -> 1 action; 2-4 -> 2; 5-6 -> 3; 7+ -> 4.
# A mode is the start_turn of the solver used for an exchange that begins in
# that regime. The exact 4-regime split is (1, 2, 5, 7); the pool takes an
# arbitrary tuple, defaulting to 3 modes (1, 3, 7) = 9 x 3 = 27 solvers.
DEFAULT_TURN_MODES = (1, 3, 7)


def make_params(hA0, hB0, start_turn, cap, stall_cap, gamma):
    return {"typeA": Type.C, "typeB": Type.C, "atkA": 2000, "atkB": 2000,
            "hpA": hA0 * 2000, "hpB": hB0 * 2000, "cap": cap,
            "stall_cap": stall_cap, "start_turn": start_turn,
            "opp_remainder": 0, "cfr_plus": True, "dtype": "float32",
            "gamma": gamma}


def build_solver(params):
    p = dict(params)
    dtype = p.pop("dtype")
    import numpy as np
    return Solver(dtype=np.float32 if dtype == "float32" else np.float64, **p)


def _train_worker(job):
    key, params, iters, report_every = job
    s = build_solver(params)
    t0 = time.time()
    for i in range(1, iters + 1):
        s.iteration()
        if report_every and i % report_every == 0:
            print("  key=%s iter %5d/%d (%.0fs, %.2f it/s)"
                  % (key, i, iters, time.time() - t0, i / (time.time() - t0)),
                  flush=True)
    s.release_graph(keep_query=False)
    return key, {"params": params, "avg": s.avg, "regret": s.regret, "_t": s._t}


class SolverPool:
    """Train (parallel) and serve (lazy-load) the 1v1 solver pool."""

    def __init__(self, out_dir, hits=(3, 4, 5), turn_modes=DEFAULT_TURN_MODES,
                 cap=20, stall_cap=3, gamma=1.0, n_jobs=6):
        self.out_dir = out_dir
        self.cap = cap
        self.stall_cap = stall_cap
        self.gamma = gamma
        self.n_jobs = n_jobs
        self.turn_modes = tuple(turn_modes)
        self.keys = [(a, b, t) for a in hits for b in hits for t in self.turn_modes]
        self._pool_path = os.path.join(out_dir, "pool.pkl")

    def _params(self, key):
        hA0, hB0, start_turn = key
        return make_params(hA0, hB0, start_turn, self.cap, self.stall_cap,
                           self.gamma)

    def train(self, iters=2000, report_every=500):
        os.makedirs(self.out_dir, exist_ok=True)
        jobs = [(key, self._params(key), iters, report_every)
                for key in self.keys]
        print("SolverPool: %d solvers x %d iters on %d workers"
              % (len(jobs), iters, self.n_jobs), flush=True)
        t0 = time.time()
        results = {}
        with Pool(self.n_jobs) as pool:
            for key, data in pool.imap_unordered(_train_worker, jobs):
                results[key] = data
                print("  done %s (%.0fs elapsed)"
                      % (key, time.time() - t0), flush=True)
        with open(self._pool_path, "wb") as fh:
            pickle.dump(results, fh)
        print("SolverPool: %d trained in %.1fs -> %s"
              % (len(results), time.time() - t0, self._pool_path), flush=True)
        return results

    def load(self):
        """Load the pool as {key: Solver} for querying. Rebuilds each solver
        from params (graph build is the cheap ~1 s at cap=8, ~30 s at cap=20)
        and restores avg/regret/_t."""
        raw = pickle.load(open(self._pool_path, "rb"))
        pool = {}
        for key, data in raw.items():
            s = build_solver(data["params"])
            s.avg = data["avg"]
            s.regret = data["regret"]
            s._t = data["_t"]
            pool[key] = s
        return pool


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"artifacts\solver_pool")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--cap", type=int, default=20)
    ap.add_argument("--gamma", type=float, default=1.0)
    a = ap.parse_args()
    pool = SolverPool(out_dir=a.out, cap=a.cap, gamma=a.gamma, n_jobs=a.jobs)
    pool.train(iters=a.iters)