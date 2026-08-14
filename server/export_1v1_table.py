"""Export a deduplicated per-state 1v1 value table from a solver checkpoint.

Writes CSV: hA,hB,bankA,bankB,shA,shB,turn,to_move,value
The table is the canonical-belief 1v1 value (the equilibrium posterior of the
1v1 solve); the belief-dependence is a known limitation (values can differ per
belief by up to ~1.9 in edge states).
"""
import os
import pickle
import sys
import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from cote_megaverse.rules import Type
from cote_megaverse.solver_tree_fh import Solver


def main():
    cap = 6
    hp = 10000
    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "server_out_fh", "ckpt_fh_cap%d.pkl" % cap)
    if not os.path.exists(ckpt):
        print("no checkpoint at %s" % ckpt)
        return
    params = {"typeA": Type.C, "typeB": Type.C, "atkA": 2000, "atkB": 2000,
              "hpA": hp, "hpB": hp, "cap": cap, "stall_cap": 3,
              "cfr_plus": True, "dtype": "float32", "gamma": 0.995,
              "full_history": True}
    data = pickle.load(open(ckpt, "rb"))
    s = Solver(**params)
    s.avg = data["avg"]
    s.regret = data["regret"]
    s._t = data["_t"]

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

    seen = {}
    for i in range(n):
        if s.term[i] is not None:
            continue
        st = s.states[i]
        vi = float(v[i])
        if st in seen:
            seen[st][0] = min(seen[st][0], vi)
            seen[st][1] = max(seen[st][1], vi)
        else:
            seen[st] = [vi, vi]

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "cote_cfr", "1v1_table_cap%d.csv" % cap)
    with open(out, "w") as fh:
        fh.write("hA,hB,bankA,bankB,shA,shB,turn,to_move,value\n")
        for st, (lo, hi) in sorted(seen.items()):
            fh.write("%d,%d,%d,%d,%d,%d,%d,%d,%.6f\n"
                     % (st[0], st[1], st[2], st[3], st[4], st[5], st[6],
                        st[7], lo))
    spread = max(hi - lo for lo, hi in seen.values()) if seen else 0.0
    print("wrote %d states -> %s (max spread %.3f)" % (len(seen), out, spread))


if __name__ == "__main__":
    main()
