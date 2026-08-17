"""Exact 2v2 ground-truth solver (Step A).

Perfect-information minimax over a tiny 2v2 subset per the official spec:
- team-wide bank (0..4), persists across switches/KOs
- fighter-local shields (turn-local; bench fighters always 0 shields)
- switch costs 1 energy, incoming allocates E-1, outgoing bench with 0 shields
- damage = rules.py rounded_damage (round-half-to-even)
- LIFO promotion on KO; terminal +1/-1 when a whole team is exhausted
- type multipliers 1.3/0.7/1.0

Solves by memoized alpha-beta (perfect info => well-defined deterministic value).
Usage:
    python server/gold_2v2.py            # tiny 2v2, 2 hits each, bank 0..1, switch allowed
"""
import os
import sys
from functools import lru_cache

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

from cote_megaverse.rules import Type, base_budget  # noqa: E402

MAX_BANK = 4
TURN_CAP = 24


def rounded_damage(atk, mult):
    return round(atk * mult / 100) * 100


def mult(a, d):
    """type multiplier: attacker type a vs defender type d"""
    if (a + 1) % 4 == d:
        return 1.3
    if (d + 1) % 4 == a:
        return 0.7
    return 1.0


def hits(hp, atk, m):
    dmg = rounded_damage(atk, m)
    return max(1, -(-hp // dmg)) if dmg else 99


# state: (turn, to_move, bankA, bankB, activeA, activeB,
#         hpA0,hpA1, atkA0,atkA1, typeA0,typeA1, shA,
#         hpB0,hpB1, atkB0,atkB1, typeB0,typeB1, shB)
# active index: 0 = fighter0, 1 = fighter1 (the other is on bench, LIFO top)
N = 2


def encode(turn, mv, bankA, bankB, actA, actB,
           hpA, atkA, tA, shA, hpB, atkB, tB, shB):
    return (turn, mv, bankA, bankB, actA, actB,
            hpA[0], hpA[1], atkA[0], atkA[1], tA[0], tA[1], shA,
            hpB[0], hpB[1], atkB[0], atkB[1], tB[0], tB[1], shB)


def make_state(turn, mv, bankA, bankB, actA, actB, hpA, atkA, tA, shA,
               hpB, atkB, tB, shB):
    return encode(turn, mv, bankA, bankB, actA, actB,
                  hpA, atkA, tA, shA, hpB, atkB, tB, shB)


def alive(hp):
    return [i for i in range(N) if hp[i] > 0]


def legal_actions(turn, mv, bankA, bankB, actA, actB, hpA, hpB):
    if mv == 0:
        e = min(8, base_budget(turn) + bankA)
        bench = [i for i in alive(hpA) if i != actA]
    else:
        e = min(8, base_budget(turn) + bankB)
        bench = [i for i in alive(hpB) if i != actB]
    acts = []
    # stay: a+d+b == e
    for a in range(e + 1):
        for d in range(e - a + 1):
            b = e - a - d
            if b <= MAX_BANK:
                acts.append((None, a, d, b))
    # switch: 1+a+d+b == e
    for k in bench:
        for a in range(e):  # 1 for switch
            for d in range(e - a):
                b = e - 1 - a - d
                if b <= MAX_BANK:
                    acts.append((k, a, d, b))
    return acts


def apply(state, action):
    (turn, mv, bankA, bankB, actA, actB,
     hpA0, hpA1, atkA0, atkA1, tA0, tA1, shA,
     hpB0, hpB1, atkB0, atkB1, tB0, tB1, shB) = state
    hpA = [hpA0, hpA1]; atkA = [atkA0, atkA1]; tA = [tA0, tA1]
    hpB = [hpB0, hpB1]; atkB = [atkB0, atkB1]; tB = [tB0, tB1]
    sw, a, d, b = action
    if mv == 0:  # A acts
        # switch phase: old active to bench with 0 shields; incoming new active
        if sw is not None:
            actA, shA = sw, 0
        # A spends b from team bank
        bankA = min(MAX_BANK, bankA + b)
        # shields: A sets d (fighter-local); bank taken from E
        shA = d
        # combat: A attacks B
        m = mult(tA[actA], tB[actB])
        dmg = rounded_damage(atkA[actA], m)
        blocked = min(shB, a)
        landed = a - blocked
        hpB[actB] -= landed * dmg
        shB = 0  # B's shields consumed
        # LIFO promotion
        if hpB[actB] <= 0 and alive(hpB):
            actB = [i for i in alive(hpB) if i != actB][0]
            shB = 0
        turn, mv = turn + 1, 1
    else:  # B acts (mirror)
        if sw is not None:
            actB, shB = sw, 0
        bankB = min(MAX_BANK, bankB + b)
        shB = d
        m = mult(tB[actB], tA[actA])
        dmg = rounded_damage(atkB[actB], m)
        blocked = min(shA, a)
        landed = a - blocked
        hpA[actA] -= landed * dmg
        shA = 0
        if hpA[actA] <= 0 and alive(hpA):
            actA = [i for i in alive(hpA) if i != actA][0]
            shA = 0
        turn, mv = turn + 1, 0
    return make_state(turn, mv, bankA, bankB, actA, actB,
                      hpA, atkA, tA, shA, hpB, atkB, tB, shB)


def terminal(state):
    _, _, _, _, actA, actB, hpA0, hpA1, *_ = state
    hpA = [hpA0, hpA1]
    hpB = [state[13], state[14]]
    a_dead = all(h <= 0 for h in hpA)
    b_dead = all(h <= 0 for h in hpB)
    if b_dead:
        return 1.0
    if a_dead:
        return -1.0
    return None


@lru_cache(maxsize=None)
def solve(state):
    t = terminal(state)
    if t is not None:
        return t
    turn, mv, bankA, bankB, actA, actB = state[0], state[1], state[2], state[3], state[4], state[5]
    hpA = [state[6], state[7]]
    hpB = [state[13], state[14]]
    if turn >= TURN_CAP:
        return 0.0
    acts = legal_actions(turn, mv, bankA, bankB, actA, actB, hpA, hpB)
    if not acts:
        return 0.0
    if mv == 0:  # A maximizes
        best = -2.0
        for act in acts:
            v = solve(apply(state, act))
            if v > best:
                best = v
            if best >= 1.0:
                break
        return best
    best = 2.0
    for act in acts:
        v = solve(apply(state, act))
        if v < best:
            best = v
        if best <= -1.0:
            break
    return best


# ---- Bench-Aware Sequential Matchup Chain DP (leaf oracle) -----------------

_TABLE = None


def _load_table():
    global _TABLE
    if _TABLE is None:
        _TABLE = {}
        with open(os.path.join(BASE, "cote_cfr", "1v1_table.csv")) as fh:
            for row in fh:
                if row.startswith("hA"):
                    continue
                p = row.strip().split(",")
                if len(p) < 8 or int(p[6]) != 7:
                    continue
                _TABLE[(int(p[0]), int(p[1]), int(p[2]), int(p[3]),
                        int(p[4]), int(p[5]))] = float(p[7])
    return _TABLE


def v1(hA, hB, mv, bk=0, sh=0, R=0):
    t = _load_table().get((min(hA, 16), min(hB, 16), mv, bk, sh, R))
    return t if t is not None else 0.0


def p_win(hA, hB, mv, rho):
    """prob the A-side active wins the duel vs hB; residual after win uses rho."""
    v = v1(hA, hB, mv)
    return min(1.0, max(0.0, (v + 1.0) / 2.0)), v


def residual_after_win(h_winner, h_loser, rho):
    """expected residual hits of the duel winner (damage-exchange model)."""
    return max(1, h_winner - max(1, round(h_loser * rho)))


def chain_leaf(rosterA, rosterB, move, rho, memo=None):
    """roster: list of hits (active first, rest = LIFO bench in order).
    move: 0 = A's team acts next, 1 = B's team acts next. Returns W in [-1,1]."""
    if memo is None:
        memo = {}
    key = (tuple(rosterA), tuple(rosterB), move)
    if key in memo:
        return memo[key]
    if not rosterA:
        memo[key] = -1.0
        return -1.0
    if not rosterB:
        memo[key] = 1.0
        return 1.0
    hA, hB = rosterA[0], rosterB[0]
    # A-perspective value of the active duel with the given mover
    mv = 0 if move == 0 else 1
    pA, _ = p_win(hA, hB, mv, rho)
    if pA <= 0.0:
        # A's active cannot win -> A loses this duel, B's active continues
        hB_new = residual_after_win(hB, hA, rho)
        val = chain_leaf(rosterA[1:], [hB_new] + rosterB[1:], 0, rho, memo)
    elif pA >= 1.0:
        hA_new = residual_after_win(hA, hB, rho)
        val = chain_leaf([hA_new] + rosterA[1:], rosterB[1:], 1, rho, memo)
    else:
        hA_new = residual_after_win(hA, hB, rho)
        hB_new = residual_after_win(hB, hA, rho)
        val = (pA * chain_leaf([hA_new] + rosterA[1:], rosterB[1:], 1, rho, memo)
               + (1 - pA) * chain_leaf(rosterA[1:], [hB_new] + rosterB[1:], 0, rho, memo))
    memo[key] = val
    return val


def main():
    atk = 1500
    tA = [0, 0]; tB = [2, 2]
    hp2 = [3000, 3000]   # 2 hits each
    hp3 = [4500, 3000]   # active 3 hits, bench 2 hits
    hp_b3 = [3000, 4500]  # active 2 hits, bench 3 hits

    def st(hpA, hpB, bankA=0, bankB=0):
        return make_state(turn=7, mv=0, bankA=bankA, bankB=bankB,
                          actA=0, actB=0, hpA=hpA, atkA=[atk, atk], tA=tA,
                          shA=0, hpB=hpB, atkB=[atk, atk], tB=tB, shB=0)

    print("[2v2 gold] baseline (2h vs 2h, dead benches ~ 1v1 oracle):",
          flush=True)
    base = st([3000, 0], [3000, 0])          # both benches dead = pure 1v1
    v_base = solve(base)
    print("  dead-bench V = %+.4f" % v_base, flush=True)

    cases = [
        ("equal benches (2h/2h)", hp2, hp2),
        ("A bench dead, B bench alive(2h)", [3000, 0], hp2),
        ("A active 3h vs B active 2h, benches equal", hp3, hp2),
        ("A active 2h, B bench stronger (3h)", hp2, hp_b3),
        ("A bench stronger (3h), B bench 2h", hp_b3, hp2),
    ]
    print("[2v2 gold] bench-effect vs 1v1 oracle:", flush=True)
    for name, hpA, hpB in cases:
        v = solve(st(hpA, hpB))
        print("  %-42s V=%+.4f  (oracle-err=%.3f)" % (name, v, v - v_base),
              flush=True)

    print("[chain DP] bench-aware leaf vs exact 2v2 (rho = damage exchange):",
          flush=True)
    for rho in (1.0, 0.7, 0.5):
        print("  rho=%.1f:" % rho, flush=True)
        for name, hpA, hpB in cases:
            rosterA = [hits_for(hpA[0])]
            if hpA[1] > 0:
                rosterA.append(hits_for(hpA[1]))
            rosterB = [hits_for(hpB[0])]
            if hpB[1] > 0:
                rosterB.append(hits_for(hpB[1]))
            exact = solve(st(hpA, hpB))
            w = chain_leaf(rosterA, rosterB, move=0, rho=rho)
            print("    %-42s exact=%+.3f chain=%+.3f  err=%.3f"
                  % (name, exact, w, w - exact), flush=True)


def hits_for(hp):
    return max(1, hp // 1500)


if __name__ == "__main__":
    main()
