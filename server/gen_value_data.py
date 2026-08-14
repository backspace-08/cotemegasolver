"""ReBeL/DeepStack data pipeline: label (position, belief) -> subgame value.

Samples random 2v2/3v3 mid-game positions, builds a belief-weighted MicroTree
root (belief over the opponent's hidden bank/shield split, R = bank + sh), runs
depth-limited FH-CFR, and records the root value as the training label for the
value network.

Leaf values for iteration 0: the exact 1v1 table + material heuristic (the
current engine defaults). Later iterations pass the trained network via the
leaf override, which is exactly the ReBeL bootstrap.

Output: a pickle with lists `states` (base flat states, opponent split removed),
`belief` (normalized weight vectors over the R+1 splits), `teamA`/`teamB` and
`values`.
"""
import argparse
import os
import pickle
import random
import sys
import time
from multiprocessing import Pool

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(BASE, ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import _cote_cfr  # noqa: E402

_TABLE = os.path.join(REPO, "cote_cfr", "1v1_table_cap6.csv")
_cote_cfr.load_1v1_table(_TABLE)

_NET = None  # optional value network for leaves, set by the Pool initializer

ATK_POOL = (1900, 2000, 2100)


def avail_mem_mb():
    """Available physical RAM (MB) via Windows API; -1 if unavailable."""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return int(m.ullAvailPhys // (1024 * 1024))
    except Exception:
        return -1


def _init_worker(net_path):
    global _NET
    if net_path:
        print("worker: building ValueLeaf", flush=True)
        from server.value_leaf import ValueLeaf
        _NET = ValueLeaf(net_path)
        print("worker: ValueLeaf ready", flush=True)


def sample(rng, depth, cap):
    # Curriculum: ~50% of the data must be late-game (low-HP 1v1/2v2) positions
    # where the depth-limited tree actually reaches a kill within the horizon,
    # so the labels carry real +1/-1 ground truth instead of leaf heuristics.
    endgame = rng.random() < 0.5
    if endgame:
        na = rng.choice((1, 1, 2, 2))
        nb = rng.choice((1, 1, 2, 2))
    else:
        na = rng.choice((2, 2, 3, 3, 3))
        nb = rng.choice((2, 2, 3, 3, 3))
    team_a = [(rng.randrange(4), rng.choice(ATK_POOL), 600) for _ in range(na)]
    team_b = [(rng.randrange(4), rng.choice(ATK_POOL), 600) for _ in range(nb)]
    order_a = list(range(na))
    order_b = list(range(nb))
    if endgame:
        # active at 1-2 hits (10..400 units), bench moderate
        hp_a = [rng.randint(10, 400) if i == 0 else rng.randint(50, 600)
                for i in range(na)]
        hp_b = [rng.randint(10, 400) if i == 0 else rng.randint(50, 600)
                for i in range(nb)]
    else:
        hp_a = [rng.randint(10, 600) for _ in range(na)]
        hp_b = [rng.randint(10, 600) for _ in range(nb)]
    turn = rng.randint(2, 5)
    to_move = rng.randint(0, 1)  # symmetric: either side can be acting
    r_opp = rng.randint(0, 4)    # hidden split sum of the OPPONENT
    # The acting side's own split is known; the opponent's is hidden (belief).
    if to_move == 0:
        bankA, shA = rng.randint(0, 4), rng.randint(0, 4)
        splits = [(r_opp - sh, sh) for sh in range(r_opp + 1)]  # (bankB, shB)
        bankB, shB = splits[0]  # canonical placeholder; belief carries the split
        weights = [1.0] * len(splits)

        def root_state(bank_b, sh_b):
            return ([len(order_a)] + order_a + hp_a + [bankA, shA]
                    + [len(order_b)] + order_b + hp_b + [bank_b, sh_b, turn, to_move])

        roots = [(root_state(b, s), w) for (b, s), w in zip(splits, weights)]
    else:
        bankB, shB = rng.randint(0, 4), rng.randint(0, 4)
        splits = [(r_opp - sh, sh) for sh in range(r_opp + 1)]  # (bankA, shA)
        bankA, shA = splits[0]  # canonical placeholder; belief carries the split
        weights = [1.0] * len(splits)

        def root_state(bank_a, sh_a):
            return ([len(order_a)] + order_a + hp_a + [bank_a, sh_a]
                    + [len(order_b)] + order_b + hp_b + [bankB, shB, turn, to_move])

        roots = [(root_state(b, s), w) for (b, s), w in zip(splits, weights)]

    mt = _cote_cfr.MicroTree(team_a, team_b, roots, depth=depth, cap=cap,
                             start_turn=turn)
    override = []
    if _NET is not None:
        leaves = mt.leaf_states()
        belief_vec = [w / sum(weights) for w in weights]
        override = _NET.override(team_a, team_b, leaves, belief_vec, to_move)
    mt.solve(300, 0.995, override)
    _, _, value = mt.strategy()
    return {
        "teamA": team_a,
        "teamB": team_b,
        "orderA": order_a, "hpA": hp_a,
        "orderB": order_b, "hpB": hp_b,
        "bankA": bankA, "shA": shA,
        "bankB": bankB, "shB": shB,
        "r_opp": r_opp,
        "turn": turn, "to_move": to_move,
        "belief": [w / sum(weights) for w in weights],
        "value": float(value),
    }


def _worker(task):
    seed, depth, cap = task
    rng = random.Random(seed)
    return sample(rng, depth, cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--cap", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--net", default=None, help="value network .pt for leaves")
    ap.add_argument("--out", default=os.path.join(REPO, "table_out", "vdata1.pkl"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    t0 = time.time()
    t_last, last_i = t0, 0
    tasks = [(a.seed * 1000 + i, a.depth, a.cap) for i in range(a.n)]
    with Pool(a.workers, initializer=_init_worker, initargs=(a.net,)) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(_worker, tasks)):
            rows.append(row)
            k = i + 1
            if k % 25 == 0 or k == a.n:
                t_now = time.time()
                inst = (t_now - t_last) / (k - last_i)
                avg = (t_now - t0) / k
                print(f"  {k}/{a.n} inst={inst:.2f}s/pos avg={avg:.2f}s/pos "
                      f"availRAM={avail_mem_mb()}MB ({t_now - t0:.0f}s)",
                      flush=True)
                t_last, last_i = t_now, k
    dt = time.time() - t0
    values = [r["value"] for r in rows]
    with open(a.out, "wb") as fh:
        pickle.dump({"rows": rows, "depth": a.depth, "cap": a.cap}, fh)
    print(f"{a.n} positions -> {a.out} in {dt:.0f}s ({dt/a.n:.2f}s/pos)")
    print(f"value: min={min(values):+.3f} max={max(values):+.3f} "
          f"mean={sum(values)/len(values):+.3f} "
          f"|v|>=0.5: {sum(abs(v)>=0.5 for v in values)/len(values)*100:.0f}%")


if __name__ == "__main__":
    main()
