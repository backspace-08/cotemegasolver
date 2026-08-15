"""Bot-vs-bot A/B: live agent.py (v3) vs frozen agent_v2.py (v2).

Both sides are full Planners. Fairness: each planner sees only public info --
the driver feeds it the opponent's public moves (attacks, budget, switches)
and revealed shields via observe()/observe_shields(), and each choose() masks
the opponent's hidden shields/bank internally. Seeds are shared, seats
alternate, temperature/rng are seeded so every game is reproducible.

Measures RELATIVE strength: if v3 wins materially more than 50% (adjusted for
first-mover), the fixes genuinely improved the bot.
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

from cote_megaverse.rules import GameState, Type, apply, initial  # noqa: E402
from cote_megaverse.agent import Planner as PlannerV3  # noqa: E402
from cote_megaverse.agent_v2 import Planner as PlannerV2  # noqa: E402


def run_match(seed, new_first, left="v3", right="v2", bodies=3, depth=2,
              temp=0.12, max_nodes=2000):
    left_cls = PlannerV3 if left == "v3" else PlannerV2
    right_cls = PlannerV3 if right == "v3" else PlannerV2
    rng = random.Random(seed)
    types = list(Type)
    state = initial(tuple(rng.choice(types) for _ in range(bodies)),
                    tuple(rng.choice(types) for _ in range(bodies)), rng=rng)
    if not new_first:
        state = replace(state, player_to_move=False).prepare()
    left_pl = left_cls(depth=depth, temperature=temp,
                       rng=random.Random(seed * 1000003 + 17),
                       max_nodes=max_nodes)
    right_pl = right_cls(depth=depth, temperature=temp,
                         rng=random.Random(seed * 2000003 + 17),
                         max_nodes=max_nodes)
    for _ in range(60):
        if state.player.lost or state.opponent.lost:
            break
        if state.player_to_move:
            before = state
            planning = GameState(state.player, state.opponent, state.turn, True)
            move = left_pl.choose(planning)
            state = apply(state, move)
            if move.attacks:
                left_pl.observe_attack(move.attacks,
                                       min(move.attacks, before.opponent.shields))
            right_pl.observe(move.attacks, move.bonuses, move.switch,
                             budget=before.player.actions)
        else:
            before = state
            planning = GameState(state.opponent, state.player, state.turn, True)
            move = right_pl.choose(planning)
            state = apply(state, move)
            if move.attacks:
                right_pl.observe_attack(move.attacks,
                                        min(move.attacks, before.player.shields))
            left_pl.observe(move.attacks, move.bonuses, move.switch,
                            budget=before.opponent.actions)
    if state.opponent.lost:
        return {"seed": seed, "new_first": new_first, "winner": "LEFT"}
    if state.player.lost:
        return {"seed": seed, "new_first": new_first, "winner": "RIGHT"}
    return {"seed": seed, "new_first": new_first, "winner": "DRAW"}


def _worker(task):
    seed, new_first, left, right, bodies, max_nodes = task
    return run_match(seed, new_first, left=left, right=right, bodies=bodies,
                     max_nodes=max_nodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, default=1100)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="v3vsV2")
    ap.add_argument("--left", default="v3", choices=["v2", "v3"])
    ap.add_argument("--right", default="v2", choices=["v2", "v3"])
    ap.add_argument("--bodies", type=int, default=3)
    ap.add_argument("--max-nodes", type=int, default=2000)
    a = ap.parse_args()
    tasks = [(a.seed_start + i, (i % 2 == 0), a.left, a.right, a.bodies,
              a.max_nodes) for i in range(a.games)]
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
    print(f"tag={a.tag}  match {a.left}-vs-{a.right}  games={n}  "
          f"LEFT={lw} RIGHT={rw} DRAW={dr}  LEFT-winrate={100.0*lw/n:.1f}%")
    for seat in (True, False):
        rs = [r for r in results if r["new_first"] == seat]
        wn = sum(1 for r in rs if r["winner"] == "LEFT")
        label = "LEFT-first" if seat else "RIGHT-first"
        print(f"  {label}: LEFT-wins {wn}/{len(rs)} = {100.0*wn/len(rs):.1f}%")
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
