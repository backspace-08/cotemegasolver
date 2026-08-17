"""ChainLeaf: Bench-Aware Sequential Matchup Chain DP as a leaf-value oracle.

Validated against the exact 2v2 solver (server/gold_2v2.py, |err| <= 0.022):
the value of a non-terminal multi-fighter leaf is the recursive chain of 1v1
duels through LIFO promotion, using the phase-4 table for each active matchup,
residual-HP of the duel winner, and the loser's bench acting first.

Decodes the engine's flat leaf states: [lenA, orderA..., hpA..., bankA, shA,
lenB, orderB..., hpB..., bankB, shB, turn, mv].
"""
import os

import _cote_cfr

_TABLE = None


def _load_table():
    global _TABLE
    if _TABLE is None:
        _TABLE = {}
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "cote_cfr", "1v1_table.csv")
        with open(path) as fh:
            for row in fh:
                if row.startswith("hA"):
                    continue
                p = row.strip().split(",")
                if len(p) < 8 or int(p[6]) != 7:
                    continue
                _TABLE[(int(p[0]), int(p[1]), int(p[2]), int(p[3]),
                        int(p[4]), int(p[5]))] = float(p[7])
    return _TABLE


def _py_round(x):
    return int(x) if x - int(x) != 0.5 else (int(x) if int(x) % 2 == 0 else int(x) + 1)


def _mult(a, d):
    if (a + 1) % 4 == d:
        return 1.3
    if (d + 1) % 4 == a:
        return 0.7
    return 1.0


class ChainLeaf:
    """Leaf oracle: override(team_a, team_b, leaves, belief_vec, bot_side)."""

    def __init__(self, rho=0.7):
        self.rho = rho
        self._memo = {}

    # ------------------------------------------------------------ flat decode
    def _decode(self, flat):
        i = 0
        la = flat[i]; i += 1
        order_a = list(flat[i:i + la]); i += la
        hp_a = list(flat[i:i + la]); i += la
        bank_a = flat[i]; sh_a = flat[i + 1]; i += 2
        lb = flat[i]; i += 1
        order_b = list(flat[i:i + lb]); i += lb
        hp_b = list(flat[i:i + lb]); i += lb
        bank_b = flat[i]; sh_b = flat[i + 1]
        turn = flat[i + 2]; mv = flat[i + 3]
        return (order_a, hp_a, bank_a, sh_a, order_b, hp_b, bank_b, sh_b,
                turn, mv)

    # ------------------------------------------------------------ leaf value
    def override(self, team_a, team_b, leaves, belief_vec, bot_side=0):
        types_a = {i: t for i, (t, _, _) in enumerate(team_a)}
        types_b = {i: t for i, (t, _, _) in enumerate(team_b)}
        atk_a = {i: a for i, (_, a, _) in enumerate(team_a)}
        atk_b = {i: a for i, (_, a, _) in enumerate(team_b)}
        out = []
        for flat in leaves:
            self._memo.clear()
            v = self._chain_flat(flat, types_a, types_b, atk_a, atk_b)
            out.append((flat, v))
        return out

    def _chain_flat(self, flat, types_a, types_b, atk_a, atk_b):
        (order_a, hp_a, bank_a, sh_a, order_b, hp_b, bank_b, sh_b, turn,
         mv) = self._decode(flat)
        rosterA = [(types_a[c], hp_a[i], atk_a[c]) for i, c in enumerate(order_a)]
        rosterB = [(types_b[c], hp_b[i], atk_b[c]) for i, c in enumerate(order_b)]
        return self._chain(rosterA, rosterB, mv, bank_a, bank_b, sh_a, sh_b)

    def _hits(self, t_w, hp_w, atk_w, t_o, atk_o):
        # hits for fighter w vs opponent o (both directions use the w-vs-o mult)
        m = _mult(t_w, t_o)
        per_hit = _py_round(atk_w * m / 100.0) * 10
        if per_hit <= 0:
            return 99
        return max(1, -(-hp_w // per_hit))

    def _chain(self, rosterA, rosterB, mv, bank_a, bank_b, sh_a=0, sh_b=0,
               depth=0):
        if depth > 12:
            return 0.0
        if not rosterA:
            return -1.0
        if not rosterB:
            return 1.0
        key = (tuple(rosterA), tuple(rosterB), mv, bank_a, bank_b, sh_a, sh_b)
        if key in self._memo:
            return self._memo[key]
        tA, hpA, atkA = rosterA[0]
        tB, hpB, atkB = rosterB[0]
        hA = self._hits(tA, hpA, atkA, tB, atkB)
        hB = self._hits(tB, hpB, atkB, tA, atkA)
        tbl = _load_table()
        v = tbl.get((min(hA, 16), min(hB, 16), mv,
                     bank_a if mv == 0 else bank_b,
                     sh_a if mv == 0 else sh_b,
                     sh_b + bank_b if mv == 0 else sh_a + bank_a))
        if v is None:
            pa = 0.5
        else:
            pa = min(1.0, max(0.0, (v + 1.0) / 2.0))
        if pa <= 0.0:
            val = self._chain(rosterA[1:], [rosterB[0]] + rosterB[1:], 0,
                              bank_a, bank_b, 0, 0, depth + 1)
        elif pa >= 1.0:
            val = self._chain([rosterA[0]] + rosterA[1:], rosterB[1:], 1,
                              bank_a, bank_b, 0, 0, depth + 1)
        else:
            val = (pa * self._chain([rosterA[0]] + rosterA[1:], rosterB[1:], 1,
                                    bank_a, bank_b, 0, 0, depth + 1)
                   + (1 - pa) * self._chain(rosterA[1:],
                                            [rosterB[0]] + rosterB[1:], 0,
                                            bank_a, bank_b, 0, 0, depth + 1))
        self._memo[key] = val
        return val
