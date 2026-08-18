"""Benchmark the CFR solver against each archetype (8 workers).

Runs CFRBot vs each archetype over N games (both seats, same seed split as
cfr_vs_planner). Reports CFR winrate / archetype winrate / draws / half-turns /
latency per archetype. This shows which pure styles the solver handles easily
and which create problems (exploitation weak spots).

Usage: python server/bench_cfr_vs_archetypes.py --games 40 [--workers 8]
"""
import argparse
import os
import random
import sys
import time
from dataclasses import replace
from multiprocessing import Pool

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))
sys.path.insert(0, os.path.join(BASE, "server"))

from cote_megaverse.rules import (GameState, Type, apply, initial)  # noqa: E402
from cote_megaverse.cfr_bot import CFRBot  # noqa: E402
from archetypes import ARCHETYPES  # noqa: E402


def run_match(name, seed, new_first, cfr_depth=3, cfr_iters=80, cfr_cap=6):
    rng = random.Random(seed)
    types = list(Type)
    state = initial(tuple(rng.choice(types) for _ in range(3)),
                    tuple(rng.choice(types) for _ in range(3)), rng=rng)
    if not new_first:
        state = replace(state, player_to_move=False).prepare()
    cfr = CFRBot(depth=cfr_depth, iters=cfr_iters, cap=cfr_cap,
                 compress=True, rng=random.Random(seed * 1000003 + 17))
    opp = ARCHETYPES[name](rng=random.Random(seed * 1000003 + 19))
    lat = []
    for _ in range(80):
        if state.player.lost or state.opponent.lost:
            break
        if state.player_to_move:  # CFR
            before = state
            planning = GameState(state.player, state.opponent, state.turn, True)
            t0 = time.time()
            move = cfr.choose(planning)
            lat.append((time.time() - t0) * 1000)
            state = apply(state, move)
            cfr.observe_shields(before.opponent.shields)
            opp.observe(move.attacks, move.bonuses, move.switch,
                        budget=before.player.actions, turn=before.turn)
        else:  # archetype
            before = state
            planning = GameState(state.opponent, state.player, state.turn, True)
            move = opp.choose(planning)
            state = apply(state, move)
            opp.observe_shields(before.player.shields)
            cfr.observe(move.attacks, move.bonuses, move.switch,
                        budget=before.opponent.actions, turn=before.turn)
    if state.opponent.lost:
        winner = "CFR"
    elif state.player.lost:
        winner = "ARCH"
    else:
        winner = "DRAW"
    return {"winner": winner, "half_turns": len(lat),
            "lat_ms": sum(lat) / len(lat) if lat else 0}


def worker(task):
    name, seed, nf, cd, ci, cc = task
    return {"name": name, "seed": seed, "new_first": nf,
            **run_match(name, seed, nf, cd, ci, cc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cfr-depth", type=int, default=3)
    ap.add_argument("--cfr-iters", type=int, default=80)
    a = ap.parse_args()
    tasks = []
    for name in ARCHETYPES:
        for i in range(a.games // 2):
            seed = 1100 + i
            for seat in (True, False):
                tasks.append((name, seed, seat, a.cfr_depth, a.cfr_iters, 6))
    with Pool(a.workers) as pool:
        results = list(pool.imap_unordered(worker, tasks))
    for name in ARCHETYPES:
        rs = [r for r in results if r["name"] == name]
        n = len(rs)
        cw = sum(1 for r in rs if r["winner"] == "CFR")
        dw = sum(1 for r in rs if r["winner"] == "DRAW")
        ht = [r["half_turns"] for r in rs]
        lat = [r["lat_ms"] for r in rs]
        print(f"CFR vs {name:9s}: CFR={cw} ARCH={n-cw-dw} DRAW={dw} "
              f"CFR-winrate={100.0*cw/n:.1f}%  avg_half={sum(ht)/n:.1f} "
              f"lat={sum(lat)/len(lat):.0f}ms")


if __name__ == "__main__":
    main()
