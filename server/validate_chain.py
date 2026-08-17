"""Validate Rust ChainLeaf (rho=0.7 vs physical exchange) on 2v2 gold cases.

2v2 states in units-of-10 HP. atk1500 neutral => per_hit 150 units => 300 units
= 2 hits. The exact minimax value for each case is recomputed by importing the
gold solver (gold_2v2) so both models are compared on identical data.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "server"))

import _cote_cfr as c
c.load_1v1_table(os.path.join(REPO, "cote_cfr", "1v1_table.csv"))

from gold_2v2 import solve, make_state  # noqa: E402

TEAM_A = [(0, 1500, 3000), (0, 1500, 3000)]  # type0: hits type1 at 1.3x
TEAM_B = [(1, 1500, 3000), (1, 1500, 3000)]  # type1: hits type0 at 0.7x
# asym per_hit (units): A(0) vs B(1): A=round(1500*1.3/100)*10=200, B=round(1500*0.7/100)*10=100
ASYM_PER_HIT = (200, 100)


def enc(hpA, hpB, bank_a=0, bank_b=0):
    """order contains ONLY alive fighters (as the engine's State does - dead
    ones are removed by promotion). Active first, rest LIFO bench."""
    aliveA = [i for i, h in enumerate(hpA) if h > 0]
    aliveB = [i for i, h in enumerate(hpB) if h > 0]
    return ([len(aliveA)] + aliveA + [hpA[i] for i in aliveA] + [bank_a, 0]
            + [len(aliveB)] + aliveB + [hpB[i] for i in aliveB] + [bank_b, 0]
            + [7, 0])


def gold_value(hpA, hpB):
    # exact minimax on 2v2 (units->ordinary HP x10 for the gold solver's rules)
    # gold_2v2 uses rounded_damage = round(atk*mult/100)*100 HP and hp in HP;
    # units-of-10 * 10 = HP. atk1500 neutral => 1500 HP/hit => 2 hits per 3000.
    hpA_hp = [u * 10 for u in hpA]
    hpB_hp = [u * 10 for u in hpB]
    return solve(make_state(turn=7, mv=0, bankA=0, bankB=0, actA=0, actB=0,
                            hpA=hpA_hp, atkA=[1500, 1500], tA=[0, 0], shA=0,
                            hpB=hpB_hp, atkB=[1500, 1500], tB=[2, 2], shB=0))


def main():
    cases = [
        ("2v2 equal benches (2h each)", [300, 300], [300, 300]),
        ("A bench dead (2h vs 1h+bench)", [300, 0], [300, 300]),
        ("A 3h active vs B 2h, benches equal", [450, 300], [300, 300]),
        ("A 2h vs B 3h bench", [300, 300], [300, 450]),
    ]
    # asymmetric-type cases: A type0 (1.3x vs B -> 200/hit), B type1 (0.7x -> 100/hit)
    # tiny HP to keep the exact minimax tractable: 1-2 hits each
    asym_cases = [
        ("ASYM A0 vs B1, 2h vs 2h", [400, 400], [200, 200]),    # A 2h, B 2h
        ("ASYM B strong (2h vs 1h)", [400, 400], [100, 100]),   # B needs 1 hit
        ("ASYM A strong (1h vs 2h)", [200, 200], [200, 200]),   # A kills in 1 hit
    ]
    print("[validate] Rust ChainLeaf vs exact 2v2 minimax (rho=0.7 vs physical)",
          flush=True)
    dummy = enc([300, 300], [300, 300])
    maes = {"rho": [], "phys": []}

    def run(name, hpA, hpB, teamA, teamB):
        st = enc(hpA, hpB)
        m1 = c.MicroTree(teamA, teamB, [(dummy, 1.0)], depth=0, cap=6, start_turn=7, compress=False)
        v_rho = m1.chain_value(st)
        m2 = c.MicroTree(teamA, teamB, [(dummy, 1.0)], depth=0, cap=6, start_turn=7, compress=False)
        m2.set_physical_exchange(True)
        v_phys = m2.chain_value(st)
        print("  %-40s rho=%+.3f phys=%+.3f  |delta=%.3f"
              % (name, v_rho, v_phys, abs(v_rho - v_phys)), flush=True)

    for name, hpA, hpB in cases:
        run(name, hpA, hpB, TEAM_A, TEAM_B)
    # asymmetric types: A type0 (1.3x), B type1 (0.7x) - where the formulas diverge
    for name, hpA, hpB in asym_cases:
        run(name, hpA, hpB, TEAM_A, TEAM_B)


def gold_value_asym(hpA, hpB, teamA, teamB):
    """exact minimax with asymmetric types: A type0, B type1."""
    tA = [0, 0]
    tB = [1, 1]
    hpA_hp = [u * 10 for u in hpA]
    hpB_hp = [u * 10 for u in hpB]
    return solve(make_state(turn=7, mv=0, bankA=0, bankB=0, actA=0, actB=0,
                            hpA=hpA_hp, atkA=[1500, 1500], tA=tA, shA=0,
                            hpB=hpB_hp, atkB=[1500, 1500], tB=tB, shB=0))


if __name__ == "__main__":
    main()
