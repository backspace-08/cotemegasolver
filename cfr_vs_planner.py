"""Head-to-head: CFR micro-tree bot (cfr_bot) vs the Planner (agent).

Both sides see only public info; the driver feeds the opponent's public moves
via observe()/observe_shields(). Seats alternate with shared seeds so every
game is reproducible. Measures CFR win rate vs the Planner.
"""
import argparse
import json
import os
import random
import sys
import time
from dataclasses import replace
from multiprocessing import Pool

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

from cote_megaverse.rules import GameState, Type, apply, initial  # noqa: E402
from cote_megaverse.agent import Planner  # noqa: E402
from cote_megaverse.cfr_bot import CFRBot  # noqa: E402


def run_match(seed, new_first, cfr_depth, cfr_iters, cfr_cap,
              pl_depth, pl_max_nodes, net=None):
    rng = random.Random(seed)
    types = list(Type)
    state = initial(tuple(rng.choice(types) for _ in range(3)),
                    tuple(rng.choice(types) for _ in range(3)), rng=rng)
    if not new_first:
        state = replace(state, player_to_move=False).prepare()
    value_leaf = None
    if net:
        from server.value_leaf import ValueLeaf
        value_leaf = ValueLeaf(net)
    else:
        from cote_megaverse.chain_leaf import ChainLeaf
        value_leaf = ChainLeaf()
    cfr = CFRBot(depth=cfr_depth, iters=cfr_iters, cap=cfr_cap,
                 value_leaf=value_leaf, compress=True)
    pl = Planner(depth=pl_depth, max_nodes=pl_max_nodes)
    lat = []
    try:
        for _ in range(80):
            if state.player.lost or state.opponent.lost:
                break
            if state.player_to_move:
                before = state
                planning = GameState(state.player, state.opponent, state.turn, True)
                t0 = time.time()
                move = cfr.choose(planning)
                lat.append((time.time() - t0) * 1000)
                state = apply(state, move)
                # The planner's shields the bot's attack met are revealed in
                # full on every resolution (RULES.md §8).
                cfr.observe_shields(before.opponent.shields)
                pl.observe(move.attacks, move.bonuses, move.switch,
                           budget=before.player.actions)
            else:
                before = state
                planning = GameState(state.opponent, state.player, state.turn, True)
                move = pl.choose(planning)
                state = apply(state, move)
                # The bot's shields the planner's attack met are revealed in
                # full on every resolution (RULES.md §8).
                pl.observe_shields(before.player.shields)
                cfr.observe(move.attacks, move.bonuses, move.switch,
                            budget=before.opponent.actions)
    except Exception:
        import traceback
        with open(os.path.join(BASE, "table_out", "worker_err.log"), "w") as fh:
            traceback.print_exc(file=fh)
        raise
    if state.opponent.lost:
        return {"seed": seed, "new_first": new_first, "winner": "LEFT",
                "half_turns": state.turn, "lat_ms": (sum(lat) / len(lat) if lat else 0.0),
                "max_lat_ms": max(lat) if lat else 0.0}
    if state.player.lost:
        return {"seed": seed, "new_first": new_first, "winner": "RIGHT",
                "half_turns": state.turn, "lat_ms": (sum(lat) / len(lat) if lat else 0.0),
                "max_lat_ms": max(lat) if lat else 0.0}
    return {"seed": seed, "new_first": new_first, "winner": "DRAW",
            "half_turns": state.turn, "lat_ms": (sum(lat) / len(lat) if lat else 0.0),
            "max_lat_ms": max(lat) if lat else 0.0}


def _worker(task):
    seed, new_first, cfr_depth, cfr_iters, cfr_cap, pl_depth, pl_max_nodes, net = task
    return run_match(seed, new_first, cfr_depth, cfr_iters, cfr_cap,
                     pl_depth, pl_max_nodes, net)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cfr-depth", type=int, default=3)
    ap.add_argument("--cfr-iters", type=int, default=200)
    ap.add_argument("--cfr-cap", type=int, default=6)
    ap.add_argument("--pl-depth", type=int, default=2)
    ap.add_argument("--pl-max-nodes", type=int, default=2000)
    ap.add_argument("--seed-start", type=int, default=1100)
    ap.add_argument("--net", default=None, help="value-network .pt for leaves")
    ap.add_argument("--tag", default="cfr_vs_planner")
    a = ap.parse_args()
    tasks = []
    for i in range(a.games // 2):
        seed = a.seed_start + i
        for seat in (True, False):
            tasks.append((seed, seat, a.cfr_depth, a.cfr_iters,
                          a.cfr_cap, a.pl_depth, a.pl_max_nodes, a.net))
    with Pool(a.workers) as pool:
        results = list(pool.imap_unordered(_worker, tasks))
    out = os.path.join(BASE, "runs", "botvbot", f"{a.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)
    n = len(results)
    lw = sum(1 for r in results if r["winner"] == "LEFT")
    rw = sum(1 for r in results if r["winner"] == "RIGHT")
    dr = sum(1 for r in results if r["winner"] == "DRAW")
    score = {"LEFT": 1.0, "RIGHT": -1.0, "DRAW": 0.0}
    by_seed = {}
    for r in results:
        by_seed[r["seed"]] = by_seed.get(r["seed"], 0.0) + score[r["winner"]]
    deltas = list(by_seed.values())
    ht = [r["half_turns"] for r in results]
    print(f"tag={a.tag} CFR-vs-Planner games={n} ({n//2} pairs x 2 seats)  "
          f"CFR={lw} Planner={rw} DRAW={dr}  CFR-winrate={100.0*lw/n:.1f}%")
    print(f"  mean delta per seed = {sum(deltas)/len(deltas):+.3f}  "
          f"(0=equal, +2=sweep)")
    for seat in (True, False):
        rs = [r for r in results if r["new_first"] == seat]
        wn = sum(1 for r in rs if r["winner"] == "LEFT")
        label = "CFR-first" if seat else "Planner-first"
        print(f"  {label}: CFR-wins {wn}/{len(rs)} = {100.0*wn/len(rs):.1f}%")
    if ht:
        print(f"  avg half_turns={sum(ht)/len(ht):.1f}  min={min(ht)} max={max(ht)}  "
              f"draws={100.0*dr/n:.1f}%")
    lats = [r["lat_ms"] for r in results if r.get("lat_ms")]
    if lats:
        print(f"  CFR avg lat={sum(lats)/len(lats):.0f}ms  "
              f"max lat={max(r['max_lat_ms'] for r in results):.0f}ms")
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
