"""Direct test of the new 1v1 table: CFRBot vs Planner in PURE 1v1 duels.

Adapts run_match to a 1v1 initial state (1 char per side) so the new table is
exercised every move. Seats alternate. Win rate should be high if the table
gives the bot decisive duel play.
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
    GameState, Side, Character, Type, apply)
from cote_megaverse.agent import Planner  # noqa: E402
from cote_megaverse.cfr_bot import CFRBot  # noqa: E402

HP_POOL = (5700, 5800, 5900, 6000, 6100, 6200, 6300)
ATK_POOL = (1900, 2000, 2100)


def make_side(t, rng):
    hp = rng.choice(HP_POOL)
    atk = rng.choice(ATK_POOL)
    return Side(characters=(Character(type=t, hp=hp, atk=atk, max_hp=hp),),
                active=0, stack_order=(0,), bonus=0, shields=0, actions=0)


def run_duel(seed, cfr_first, cfr_depth, cfr_iters, cfr_cap,
             pl_depth, pl_max_nodes, cfr_gamma,
             max_half_turns=60):
    rng = random.Random(seed)
    types = list(Type)
    state = GameState(player=make_side(rng.choice(types), rng),
                      opponent=make_side(rng.choice(types), rng),
                      turn=1, player_to_move=True).prepare()
    if not cfr_first:
        state = replace(state, player_to_move=False).prepare()
    cfr = CFRBot(depth=cfr_depth, iters=cfr_iters, cap=cfr_cap, gamma=cfr_gamma)
    pl = Planner(depth=pl_depth, max_nodes=pl_max_nodes)
    for _ in range(max_half_turns):
        if state.player.lost or state.opponent.lost:
            break
        if state.player_to_move:
            before = state
            planning = GameState(state.player, state.opponent, state.turn, True)
            move = cfr.choose(planning)
            state = apply(state, move)
            pl.observe(move.attacks, move.bonuses, move.switch,
                       budget=before.player.actions)
            pl.observe_shields(before.player.shields)
        else:
            before = state
            planning = GameState(state.opponent, state.player, state.turn, True)
            move = pl.choose(planning)
            state = apply(state, move)
            cfr.observe(move.attacks, move.bonuses, move.switch,
                        budget=before.opponent.actions)
            cfr.observe_shields(before.opponent.shields)
    if state.opponent.lost:
        return {"seed": seed, "cfr_first": cfr_first, "winner": "CFR",
                "half_turns": state.turn}
    if state.player.lost:
        return {"seed": seed, "cfr_first": cfr_first, "winner": "PL",
                "half_turns": state.turn}
    return {"seed": seed, "cfr_first": cfr_first, "winner": "DRAW",
            "half_turns": state.turn}


def _worker(task):
    seed, cfr_first, cfr_depth, cfr_iters, cfr_cap, pl_depth, pl_max_nodes, cfr_gamma = task
    return run_duel(seed, cfr_first, cfr_depth, cfr_iters, cfr_cap,
                    pl_depth, pl_max_nodes, cfr_gamma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cfr-depth", type=int, default=3)
    ap.add_argument("--cfr-iters", type=int, default=100,
                    help="CFR iters per re-solve (100 == 200 results, ~2x faster)")
    ap.add_argument("--cfr-cap", type=int, default=6)
    ap.add_argument("--pl-depth", type=int, default=2)
    ap.add_argument("--pl-max-nodes", type=int, default=2000)
    ap.add_argument("--cfr-gamma", type=float, default=0.995)
    ap.add_argument("--seed-start", type=int, default=7000)
    ap.add_argument("--tag", default="duel")
    a = ap.parse_args()
    tasks = [(a.seed_start + i, (i % 2 == 0), a.cfr_depth, a.cfr_iters,
              a.cfr_cap, a.pl_depth, a.pl_max_nodes, a.cfr_gamma)
             for i in range(a.games)]
    with Pool(a.workers) as pool:
        results = list(pool.imap_unordered(_worker, tasks))
    out = os.path.join(BASE, "runs", "botvbot", "%s.json" % a.tag)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)
    n = len(results)
    cw = sum(1 for r in results if r["winner"] == "CFR")
    pw = sum(1 for r in results if r["winner"] == "PL")
    dr = sum(1 for r in results if r["winner"] == "DRAW")
    lens = [r["half_turns"] for r in results]
    print("tag=%s 1v1 duels=%d  CFR=%d Planner=%d DRAW=%d  CFR-winrate=%.1f%%"
          % (a.tag, n, cw, pw, dr, 100.0 * cw / n))
    print("  avg half_turns=%.1f  min=%d max=%d  draws=%.1f%%"
          % (sum(lens) / n, min(lens), max(lens), 100.0 * dr / n))
    for seat in (True, False):
        rs = [r for r in results if r["cfr_first"] == seat]
        wn = sum(1 for r in rs if r["winner"] == "CFR")
        print("  CFR-first=%s: CFR-wins %d/%d = %.1f%%"
              % (seat, wn, len(rs), 100.0 * wn / len(rs)))
    print("results -> %s" % out)


if __name__ == "__main__":
    main()
