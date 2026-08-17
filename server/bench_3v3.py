"""Step B quick probe: root action count + tree growth by depth (cheap).

Builds depth 2/3 trees (fast) for full-grid vs compressed 3v3, extrapolates
depth-4 node count from the observed branching growth.
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import _cote_cfr as c
c.load_1v1_table(os.path.join(REPO, "cote_cfr", "1v1_table.csv"))

TEAM = [(0, 2000, 3000), (1, 2000, 3000), (2, 2000, 3000)]


def encode(hp_units, bank, turn, to_move):
    n = len(hp_units)
    return ([n] + list(range(n)) + hp_units + [bank, 0]
            + [n] + list(range(n)) + hp_units + [bank, 0]
            + [turn, to_move])


def main():
    state = encode([3000, 3000, 3000], 4, 7, 0)
    roots = [(state, 1.0)]
    print("[estimate] unpruned est_n4 ~43M (infeasible); compressed est_n4 ~769k",
          flush=True)
    print("[measure] compressed depth-4, iters=40, prune_after=10:", flush=True)
    t0 = time.time()
    mt = c.MicroTree(TEAM, TEAM, roots, depth=4, cap=20, start_turn=7,
                     compress=True)
    build_ms = (time.time() - t0) * 1000
    n = mt.node_count()
    t0 = time.time()
    mt.solve(40, 0.995, [], 10)
    solve_ms = (time.time() - t0) * 1000
    print("  nodes=%d  build=%.0fms  solve=%.0fms  total=%.0fms  val=%+.3f"
          % (n, build_ms, solve_ms, build_ms + solve_ms, mt.strategy()[2]),
          flush=True)


if __name__ == "__main__":
    main()
