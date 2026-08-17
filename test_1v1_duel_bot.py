"""CFRBot vs Planner in PURE 1v1 duels - stat-profile benchmark.

Design (3 requirements):
1. Stat profiles sized for the 1..16 hits grid so both sides get 2-3 full
   burst cycles. hits = ceil(HP / per_hit), per_hit = round(ATK*mult/100)*100.
     A. deep-midgame (main): HP=24000 ATK=1500, 1.0x vs 1.0x -> 16 vs 16 hits
     B. cfr-defense  (stress): HP=22000 ATK=2000, CFR 0.7x vs PL 1.3x
        -> Planner kills CFR in 9 hits, CFR needs 16 -> physical handicap
     C. cfr-attack   (exploit): HP=22000 ATK=2000, CFR 1.3x vs PL 0.7x
        -> CFR kills Planner in 9 hits, survives 16 -> forced-lethal speed
2. Spam/draw isolation: a4-shield spam ends in a draw at the turn cap (0.0),
   never a win; scoring is Win=+1.0, Draw=0.0, Loss=-1.0. Turn cap 40
   half-turns covers 3 burst exchanges of 16 hits.
3. Duplicate Matching: each seed runs twice (CFR as P1 and as P2); per-seed
   delta = score(P1) + score(P2) cancels the first-move advantage.
   delta=0 -> equal, >0 -> CFR stronger, +2 -> sweep.

Default: 50 duplicate pairs (100 matches) on Profile A.
"""
import argparse
import json
import os
import random
import sys
from dataclasses import replace
from multiprocessing import Pool

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

from cote_megaverse.rules import (  # noqa: E402
    GameState, Side, Character, Type, apply, per_hit_damage, multiplier)
from cote_megaverse.agent import Planner  # noqa: E402
from cote_megaverse.cfr_bot import CFRBot  # noqa: E402

WIN, DRAW, LOSS = 1.0, 0.0, -1.0

# type value layout: A=0 B=1 C=2 D=3; (attacker+1)%4==defender -> 1.3
BEATS = {Type.A: Type.B, Type.B: Type.C, Type.C: Type.D, Type.D: Type.A}

PROFILES = {
    "A": {"hp": 24000, "atk": 1500,
          "pl_type": Type.A, "cfr_type": Type.A},          # 1.0 vs 1.0 -> 16v16
    "B": {"hp": 22000, "atk": 2000,
          "pl_type": Type.D, "cfr_type": Type.A},          # D beats A: PL 1.3
    "C": {"hp": 22000, "atk": 2000,
          "pl_type": Type.B, "cfr_type": Type.A},          # A beats B: CFR 1.3
    "E": {"even": "E", "atk": 1500, "pl_type": Type.A, "cfr_type": Type.A},
    "F": {"even": "F", "atk": 1500, "pl_type": Type.A, "cfr_type": Type.A},
    "G": {"even": "G", "atk": 1500, "pl_type": Type.A, "cfr_type": Type.A},
    "I": {"even": "I", "atk": 1500, "pl_type": Type.A, "cfr_type": Type.A},
}


def even_seat(profile, cfr_first):
    """(cfr_hits, pl_hits, cfr_hidden, pl_hidden) for a near-even start, seat by seat.
    hidden = (shields, bank) the side currently holds (opponent's R is hidden)."""
    if profile == "E":      # (13,16) A-to-move, clean: fewer-hit side opens
        return (13, 16, (0, 0), (0, 0)) if cfr_first else (16, 13, (0, 0), (0, 0))
    if profile == "F":      # (11,14)
        return (11, 14, (0, 0), (0, 0)) if cfr_first else (14, 11, (0, 0), (0, 0))
    if profile == "G":      # (12,15)
        return (12, 15, (0, 0), (0, 0)) if cfr_first else (15, 12, (0, 0), (0, 0))
    if profile == "I":      # (13,15) turn-1 even (calibrated at start-turn 1)
        return (13, 15, (0, 0), (0, 0)) if cfr_first else (15, 13, (0, 0), (0, 0))
    raise ValueError(profile)


def profile_hits(prof):
    """(hits CFR needs to kill PL, hits PL needs to kill CFR) from the formula."""
    def per_hit(atk, mult):
        return per_hit_damage(
            Character(type=Type.A, hp=1, atk=atk, max_hp=1),
            Character(type=Type.A, hp=1, atk=1, max_hp=1)) if mult == 1.0 else \
            round(atk * mult / 100) * 100
    cfr_m = multiplier(prof["cfr_type"], prof["pl_type"])
    pl_m = multiplier(prof["pl_type"], prof["cfr_type"])
    h = prof["hp"]
    return (max(1, -(-h // per_hit(prof["atk"], cfr_m))),
            max(1, -(-h // per_hit(prof["atk"], pl_m))))


def make_side(t, hp, atk, shields=0, bonus=0):
    return Side(characters=(Character(type=t, hp=hp, atk=atk, max_hp=hp),),
                active=0, stack_order=(0,), bonus=bonus, shields=shields,
                actions=0)


def run_duel(seed, cfr_first, profile, cfr_depth, cfr_iters, cfr_cap,
             pl_depth, pl_max_nodes, cfr_gamma, cfr_prune_after,
             cfr_temperature, start_turn, max_half_turns=40):
    prof = PROFILES[profile]
    if "even" in prof:
        cfr_h, pl_h, cfr_hid, pl_hid = even_seat(prof["even"], cfr_first)
        cfr_hp = cfr_h * 1500
        pl_hp = pl_h * 1500
        cfr_sh, cfr_bk = cfr_hid
        pl_sh, pl_bk = pl_hid
    else:
        cfr_hp = pl_hp = prof["hp"]
        cfr_sh = cfr_bk = pl_sh = pl_bk = 0
    state = GameState(player=make_side(prof["cfr_type"], cfr_hp, prof["atk"],
                                       shields=cfr_sh, bonus=cfr_bk),
                      opponent=make_side(prof["pl_type"], pl_hp, prof["atk"],
                                         shields=pl_sh, bonus=pl_bk),
                      turn=start_turn, player_to_move=True).prepare()
    if not cfr_first:
        state = replace(state, player_to_move=False).prepare()
    cfr = CFRBot(depth=cfr_depth, iters=cfr_iters, cap=cfr_cap, gamma=cfr_gamma,
                 prune_after=cfr_prune_after, temperature=cfr_temperature,
                 rng=random.Random(seed))
    pl = Planner(depth=pl_depth, max_nodes=pl_max_nodes)
    for _ in range(max_half_turns):
        if state.player.lost or state.opponent.lost:
            break
        if state.player_to_move:
            before = state
            planning = GameState(state.player, state.opponent, state.turn, True)
            move = cfr.choose(planning)
            state = apply(state, move)
            cfr.observe_shields(before.opponent.shields)
            pl.observe(move.attacks, move.bonuses, move.switch,
                       budget=before.player.actions)
        else:
            before = state
            planning = GameState(state.opponent, state.player, state.turn, True)
            move = pl.choose(planning)
            state = apply(state, move)
            pl.observe_shields(before.player.shields)
            cfr.observe(move.attacks, move.bonuses, move.switch,
                        budget=before.opponent.actions)
    if state.opponent.lost:
        return {"seed": seed, "cfr_first": cfr_first, "winner": "CFR",
                "score": WIN, "half_turns": state.turn}
    if state.player.lost:
        return {"seed": seed, "cfr_first": cfr_first, "winner": "PL",
                "score": LOSS, "half_turns": state.turn}
    return {"seed": seed, "cfr_first": cfr_first, "winner": "DRAW",
            "score": DRAW, "half_turns": state.turn}


def _worker(task):
    (seed, cfr_first, profile, cfr_depth, cfr_iters, cfr_cap, pl_depth,
     pl_max_nodes, cfr_gamma, cfr_prune_after, cfr_temperature,
     start_turn) = task
    return run_duel(seed, cfr_first, profile, cfr_depth, cfr_iters, cfr_cap,
                    pl_depth, pl_max_nodes, cfr_gamma, cfr_prune_after,
                    cfr_temperature, start_turn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=sorted(PROFILES), default="A")
    ap.add_argument("--pairs", type=int, default=50,
                    help="duplicate seed pairs; 2 matches per pair (P1+P2) = 100")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-half-turns", type=int, default=40,
                    help="turn cap; reaching it is a draw (0.0)")
    ap.add_argument("--cfr-depth", type=int, default=3,
                    help="phase-4 (turn>=7) resolve depth; even depth = whole rounds")
    ap.add_argument("--cfr-iters", type=int, default=120)
    ap.add_argument("--cfr-prune-after", type=int, default=20)
    ap.add_argument("--cfr-temperature", type=float, default=1.0,
                    help="sample actions from the CFR strategy (1.0); 0 = argmax")
    ap.add_argument("--cfr-cap", type=int, default=6)
    ap.add_argument("--pl-depth", type=int, default=2)
    ap.add_argument("--pl-max-nodes", type=int, default=2000)
    ap.add_argument("--cfr-gamma", type=float, default=0.995)
    ap.add_argument("--seed-start", type=int, default=7000)
    ap.add_argument("--start-turn", type=int, default=7,
                    help="duel starts at this turn (7 = phase-4, table is authoritative)")
    ap.add_argument("--tag", default="duel")
    a = ap.parse_args()

    prof = PROFILES[a.profile]
    if "even" in prof:
        pos = {  # noqa: E702
            "E": "(13,16) R0 V=-0.041 @t7", "F": "(11,14) R0 V=-0.038 @t7",
            "G": "(12,15) R0 V=-0.038 @t7",
            "I": "(13,15) R0 V=-0.068 @t1",
        }[prof["even"]]
        print("[profile %s] even %s, 13-hit style side opens" % (a.profile, pos),
              flush=True)
        if prof["even"] == "I" and a.start_turn != 1:
            print("[warn] profile I is calibrated at start-turn 1; use --start-turn 1",
                  flush=True)
    else:
        hits = profile_hits(prof)
        print("[profile %s] HP=%d ATK=%d  hits: CFR->PL %d, PL->CFR %d"
              % (a.profile, prof["hp"], prof["atk"], hits[0], hits[1]), flush=True)

    tasks = []
    for i in range(a.pairs):
        seed = a.seed_start + i
        for cfr_first in (True, False):
            tasks.append((seed, cfr_first, a.profile, a.cfr_depth, a.cfr_iters,
                          a.cfr_cap, a.pl_depth, a.pl_max_nodes, a.cfr_gamma,
                          a.cfr_prune_after, a.cfr_temperature, a.start_turn))

    with Pool(a.workers) as pool:
        results = list(pool.imap_unordered(_worker, tasks))

    out = os.path.join(BASE, "runs", "botvbot", "%s.json" % a.tag)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)

    n = len(results)
    by_seat = {True: [], False: []}
    delta_by_seed = {}
    for r in results:
        by_seat[r["cfr_first"]].append(r["score"])
        delta_by_seed[r["seed"]] = delta_by_seed.get(r["seed"], 0.0) + r["score"]
    deltas = list(delta_by_seed.values())
    mean_delta = sum(deltas) / len(deltas)

    def agg(scores):
        w = sum(1 for s in scores if s > 0)
        d = sum(1 for s in scores if s == 0)
        l = sum(1 for s in scores if s < 0)
        return w, d, l

    cw, cd, cl = agg([r["score"] for r in results])
    fw, fd, fl = agg(by_seat[True])
    sw, sd, sl = agg(by_seat[False])
    lens = [r["half_turns"] for r in results]

    print("[profile %s] duels=%d (%d pairs x 2 seats)  "
          "CFR=%d Planner=%d DRAW=%d  CFR-winrate=%.1f%%" % (
              a.profile, n, a.pairs, cw, cl, cd, 100.0 * cw / n))
    print("  mean delta per seed = %+.3f  (0=equal, +2=sweep)"
          % mean_delta, flush=True)
    print("  seeds: sweep +2=%d, +1=%d, 0=%d, -1=%d, -2=%d"
          % tuple(sum(1 for d in deltas if d == x) for x in (2, 1, 0, -1, -2)),
          flush=True)
    print("  CFR-first (P1): W%d/D%d/L%d = %.1f%%   CFR-second (P2): W%d/D%d/L%d = %.1f%%"
          % (fw, fd, fl, 100.0 * fw / len(by_seat[True]),
             sw, sd, sl, 100.0 * sw / len(by_seat[False])), flush=True)
    print("  avg half_turns=%.1f  min=%d max=%d  draws=%.1f%%" % (
        sum(lens) / n, min(lens), max(lens), 100.0 * cd / n), flush=True)
    print("results -> %s" % out, flush=True)


if __name__ == "__main__":
    main()


