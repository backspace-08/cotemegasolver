"""Validate the Rust trunk port (cote_cfr) against the Python trunk rules.

Skipped if the ``_cote_cfr`` extension is not installed (build with
``python -m maturin build --release -m cote_cfr/Cargo.toml`` then
``pip install cote_cfr/target/wheels/*.whl``).
"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

try:
    import _cote_cfr
except ImportError:
    _cote_cfr = None

from cote_megaverse.trunk import Trunk

TEAM = ((0, 2000, 6000), (1, 2000, 6000), (2, 2100, 6300))


@unittest.skipIf(_cote_cfr is None, "_cote_cfr not built/installed")
class TestRustTrunk(unittest.TestCase):
    def test_random_walk_matches(self):
        cap = 6
        py = Trunk(TEAM, TEAM, cap=cap, start_turn=1, first_move=0)
        rs = _cote_cfr.PyTrunk(
            [(t, a, h // 10) for t, a, h in TEAM],
            [(t, a, h // 10) for t, a, h in TEAM], cap, 1)

        def to_flat(st):
            oa, ha, ba, sha, ob, hb, bb, shb, turn, tm = st
            return ([len(oa)] + [int(c) for c in oa] + [int(h) for h in ha]
                    + [ba, sha, len(ob)] + [int(c) for c in ob]
                    + [int(h) for h in hb] + [bb, shb, turn, tm])

        def from_flat(f):
            p = 0
            la = f[p]; p += 1
            oa = tuple(f[p:p + la]); p += la
            ha = tuple(f[p:p + la]); p += la
            ba = f[p]; p += 1; sha = f[p]; p += 1
            lb = f[p]; p += 1
            ob = tuple(f[p:p + lb]); p += lb
            hb = tuple(f[p:p + lb]); p += lb
            bb = f[p]; p += 1; shb = f[p]; p += 1
            turn = f[p]; p += 1; tm = f[p]
            return (oa, ha, ba, sha, ob, hb, bb, shb, turn, tm)

        def act_flat(act):
            a, d, b, sw = act
            return [a, d, b, -1 if sw is None else int(sw)]

        rng = random.Random(42)
        st = py.start_states[0]
        for _ in range(3000):
            acts_py = py.actions(st)
            acts_rs = sorted(tuple(x[:3]) + (None if x[3] < 0 else x[3],)
                             for x in rs.actions(to_flat(st)))
            self.assertEqual(sorted(acts_py), acts_rs)
            act = rng.choice(acts_py)
            child_py = py.transition(st, act)
            child_rs = rs.transition(to_flat(st), act_flat(act))
            self.assertEqual(child_py, from_flat(child_rs) if child_rs else None)
            if py.terminal(st) is None and child_py is not None:
                st = child_py
            else:
                st = rng.choice(py.start_states)


if __name__ == "__main__":
    unittest.main()
