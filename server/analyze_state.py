"""Exact manual-state CFR bot simulator with full solver diagnostics.

Constructs an arbitrary GameState from explicit character specs (type, hp,
atk, max_hp) for both sides, then steps a CFR bot (depth=3, compress) and
accepts human moves, dumping the full root strategy / belief / value on every
bot turn.

Usage:
    python server/analyze_state.py
Edit the STATE block below to match the board you want to analyze. Human moves
are fed in the BOT_MOVES / then interactive prompt.
"""
import os
import random
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

from cote_megaverse.rules import (GameState, Side, Character, Type, apply,  # noqa: E402
                                  Allocation, base_budget)
from cote_megaverse.cfr_bot import CFRBot  # noqa: E402


def Ch(t, hp, atk):
    if isinstance(t, str):
        t = Type[t]
    return Character(type=t, hp=hp, atk=atk, max_hp=hp)


# ---------------- EDIT THIS BLOCK to match the board ----------------
# Each tuple: (Type, hp, atk). Order = character index in the team.
# active = which index is active; bench = stack order for LIFO (live ones).
# bank = team bonus; shields = active's held shields; actions pre-set or auto.
STATE = {
    "turn": 6,
    "you": {
        # YOU team on turn 6 from the log:
        # idx0 B #1 dead, idx1 B Brawler 5900/2100 active, idx2 C 5800/1900
        "chars": [("B", 0, 0), ("B", 5900, 2100), ("C", 5800, 1900)],
        "active": 1,
        "bank": 0,
        "shields": 0,
    },
    "bot": {
        # BOT team on turn 6: active A Artist 6100/2100, bench D 2200/2000, A 6000/2000
        "chars": [("D", 2200, 2000), ("A", 6100, 2100), ("A", 6000, 2000)],
        "active": 1,
        "bank": 0,
        "shields": 0,
    },
    "you_move_first": False,   # t6 -> bot moves (you_move_first=False)
}
# ---------------------------------------------------------------------


def build_side(spec):
    chars = [Ch(*c) for c in spec["chars"]]
    order = tuple(i for i, c in enumerate(chars) if c.alive)
    return Side(characters=tuple(chars), active=spec["active"],
                stack_order=order, bonus=spec["bank"], shields=spec["shields"],
                actions=min(8, base_budget(0) + spec["bank"]))


def dump(bot, move, turn, label):
    r = bot.last_report
    print(f"\n=== {label} (BOT, turn {turn}) ===", flush=True)
    print(f"  bot move: a{move.attacks} d{move.defends} b{move.bonuses} sw={move.switch_to}",
          flush=True)
    print(f"  value=%+.4f  belief=%s" % (r["value"], r["belief"]), flush=True)
    top = sorted(r["root_actions"], key=lambda x: -x[1])[:6]
    print("  root:", [(tuple(a[:3]), round(p, 4)) for a, p in top], flush=True)
    # expand full strategy by expected-utility style: all with prob > 0.01
    full = sorted(r["root_actions"], key=lambda x: -x[1])
    print("  all:", ", ".join("%s=%.2f" % (tuple(a[:3]), p) for a, p in full
                              if p > 0.01), flush=True)


def main():
    st = STATE
    you = build_side(st["you"])
    bot_side = build_side(st["bot"])
    state = GameState(player=you, opponent=bot_side, turn=st["turn"],
                      player_to_move=st["you_move_first"]).prepare()
    bot = CFRBot(depth=3, iters=300, cap=8, gamma=0.995, prune_after=20,
                 compress=True, rng=random.Random(1))
    print("YOU:", [f"{i}:{c.type.name}/{c.hp}/{c.atk}" for i, c in enumerate(state.player.characters)])
    print("BOT:", [f"{i}:{c.type.name}/{c.hp}/{c.atk}" for i, c in enumerate(state.opponent.characters)])
    for turn in range(st["turn"], 20):
        if state.player.lost or state.opponent.lost:
            print("\nwinner:", "YOU" if state.opponent.lost else "BOT")
            break
        if state.player_to_move:
            # interactive human move: a,d,b
            raw = input(f"turn {turn} YOU (a d b) > ").strip()
            if raw.lower() in ("q", "quit"):
                break
            parts = [int(x) for x in raw.replace(",", " ").split()]
            a, d, b = (parts + [0, 0, 0])[:3]
            m = Allocation(a, d, b, None)
            before = state
            state = apply(state, m)
            bot.observe(m.attacks, m.bonuses, m.switch,
                        budget=before.player.actions, turn=before.turn)
            bot.observe_shields(before.opponent.shields)
            print(f"  YOU: a{m.attacks} d{m.defends} b{m.bonuses}")
        else:
            before = state
            planning = GameState(state.opponent, state.player, state.turn, True)
            move = bot.choose(planning)
            dump(bot, move, state.turn, "AI")
            state = apply(state, move)
            bot.observe_shields(before.player.shields)


if __name__ == "__main__":
    main()
