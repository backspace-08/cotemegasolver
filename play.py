"""Headless LLM-as-human harness: I play the human, Planner plays the bot.

Fair observation only: I see HP, ATK, types, actives, stacks, turn, the acting
side's public budget, and public resolutions. I never see the bot's hidden bank
or held shields — the harness enforces the Phase-0 boundary.

Usage:
  python play.py new --seed N [--ai_first|--human_first] [--temp T]
  python play.py move "a,d,b" [sw]
  python play.py view
  python play.py end
  python play.py session --games 20 [--seed S] [--temp T]
      Session mode: play N games in a row; after each game the result is
      recorded automatically (with timestamp) to session_log.json and
      winrate_log.json, then the next game starts immediately. You just play.
"""
import argparse
import datetime
import json
import os
import pickle
import random
import re
import sys

from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from cote_megaverse.agent import Planner
from cote_megaverse.interactive import (PlayerQuit, human_allocation,
                                        show_outcome, show_resolution,
                                        show_state)
from cote_megaverse.rules import (Allocation, GameState, MAX_ACTIONS,
                                  MAX_BONUS, Type, apply, attacks_to_kill,
                                  base_budget, exchange_damage, initial,
                                  multiplier)

_BASE = os.path.dirname(os.path.abspath(__file__))
RUN = "default"


def _set_run(name):
    """Select the per-run sandbox; everything (state + logs) lives in runs/<name>/."""
    global RUN
    RUN = name
    os.makedirs(os.path.join(_BASE, "runs", name), exist_ok=True)


def _state_path():
    return os.path.join(_BASE, "runs", RUN, "state.pkl")


def _session_log_path():
    return os.path.join(_BASE, "runs", RUN, "session_log.json")


def _winrate_log_path():
    return os.path.join(_BASE, "runs", RUN, "winrate_log.json")


def show_stats(run):
    """Summarize winrate_log.json for --run: W/L/D and per-seat split."""
    path = os.path.join(_BASE, "runs", run, "winrate_log.json")
    entries = _read_json(path, [])
    if not entries:
        print("No recorded games in runs/%s yet." % run)
        return
    total = len(entries)
    w = sum(1 for e in entries if e["result"] == "w")
    d = sum(1 for e in entries if e["result"] == "d")
    l = total - w - d
    def seat(name):
        s = [e for e in entries if e["seat"] == name]
        if not s:
            return "n/a"
        sw = sum(1 for e in s if e["result"] == "w")
        return "%.1f%%" % (100.0 * sw / len(s))
    print("runs/%s: %d games  W=%d L=%d D=%d  win%%=%.1f  (ai_first %s, human_first %s)"
          % (run, total, w, l, d, 100.0 * w / (total - d) if total > d else 0.0,
             seat("ai_first"), seat("human_first")))


def save(game):
    with open(_state_path(), "wb") as fh:
        pickle.dump(game, fh)


def load():
    with open(_state_path(), "rb") as fh:
        return pickle.load(fh)


def _read_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)


def render(state, game_index=None, session_total=None, history=None):
    lines = []
    banner = ""
    if game_index and session_total:
        banner = f"GAME {game_index}/{session_total}  |  "
    lines.append(f"=== {banner}TURN {state.turn} | "
                 f"{'YOUR move' if state.player_to_move else 'BOT move'} ===")
    for tag, side in (("YOU", state.player), ("BOT", state.opponent)):
        ch = side.characters[side.active]
        lines.append(f"[{tag}] active=#{side.active} {ch.type.name} "
                     f"hp={ch.hp}/{ch.max_hp} atk={ch.atk}  "
                     f"stack={list(side.normalized_order())}")
        for i, c in enumerate(side.characters):
            mark = ">" if i == side.active else " "
            lines.append(f"   {mark} #{i} {c.type.name} hp={c.hp} atk={c.atk}")
        # public info only
        if tag == "YOU":
            lines.append(f"   your shields: {side.shields}")
        # Public budget breakdown for the ACTING side only: base actions (from
        # turn) + banked bonus revealed on its own turn.
        acting = state.player if state.player_to_move else state.opponent
        if side is acting:
            base = base_budget(state.turn)
            bonus_actions = max(0, side.actions - base)
            lines.append(f"   Actions: {base} + {bonus_actions}  "
                         f"(total {side.actions})")
    if history:
        lines.append("  -- history (public) --")
        for h in history[-10:]:
            lines.append(f"    {h}")
    return "\n".join(lines)


def parse_intent(raw):
    """Tolerant intent parser. Returns (a, d, b, sw_1based|None) or None for '-'.

    Accepts positional numbers (separators , ; space /): '2,1', '2 1', '2/1',
    '2,0,1,2' (4th = 1-based switch target). Also keyword tokens: 'a2', 'd1',
    's1' (shield), 'b3', 'sw2' / 'switch 2'. Anything else is an error.
    """
    raw = (raw or "").strip()
    if not raw or raw == "-":
        return None
    toks = [t for t in re.split(r"[,;\s/]+", raw.lower()) if t]
    if all(t.isdigit() for t in toks):
        vals = [int(t) for t in toks]
        a = vals[0] if len(vals) > 0 else 0
        d = vals[1] if len(vals) > 1 else 0
        b = vals[2] if len(vals) > 2 else 0
        sw = vals[3] if len(vals) > 3 else None
        return (a, d, b, sw)
    a = d = b = 0
    sw = None
    for tok in toks:
        if tok.startswith("sw") or tok.startswith("switch"):
            rest = tok[2:] if tok.startswith("sw") else tok[6:]
            if rest.isdigit():
                sw = int(rest)
            else:
                raise SystemExit(f"bad switch target: {raw!r}")
        elif tok.startswith("a"):
            a = int(tok[1:])
        elif tok.startswith("d") or tok.startswith("s"):
            d = int(tok[1:])
        elif tok.startswith("b"):
            b = int(tok[1:])
        else:
            raise SystemExit(f"cannot parse move: {raw!r}")
    return (a, d, b, sw)


def _bot_noatk_streak(history):
    """Consecutive recent bot turns without an attack (the 'wall' signal)."""
    n = 0
    for h in reversed(history or []):
        if "BOT" not in h:
            continue
        if "did not attack" in h:
            n += 1
        else:
            break
    return n


def _side_dict(side, budget=None, reveal_private=False):
    ch = side.characters[side.active]
    d = {"a": side.active, "T": ch.type.name, "hp": ch.hp, "atk": ch.atk,
         "alive": [[i, c.type.name, c.hp, c.atk]
                   for i, c in enumerate(side.characters)]}
    if budget is not None:
        d["bud"] = budget
    if reveal_private:
        d["bank"] = side.bonus
        d["sh"] = side.shields
    return d


def compact_render(game):
    """One-turn decision payload: everything the player needs + precomputed
    hints (kill thresholds, worst bot reply, shields to survive, switches)."""
    st = game["state"]
    your, bot = st.player, st.opponent
    ya = your.characters[your.active]
    ba = bot.characters[bot.active]
    sbot = game.get("sBot", 0)
    mult = multiplier(ya.type, ba.type)
    nxt = min(MAX_ACTIONS, base_budget(st.turn + 1) + MAX_BONUS)
    worst = exchange_damage(ba, ya, nxt) if nxt > 0 else 0
    shld = None
    for d in range(0, nxt + 1):
        if exchange_damage(ba, ya, nxt - d) < ya.hp:
            shld = d
            break
    sw_opts = [[i + 1, c.type.name, multiplier(c.type, ba.type),
                exchange_damage(c, ba, 1)]
               for i, c in enumerate(your.characters)
               if i != your.active and c.alive]
    hint = {"mult": mult,
            "kill0": attacks_to_kill(ya, ba, 0),
            "killS": attacks_to_kill(ya, ba, sbot),
            "sBot": sbot,
            "worst": worst,
            "shld": shld,
            "sw": sw_opts,
            "streak": _bot_noatk_streak(game.get("history"))}
    obj = {"t": st.turn,
           "turn": "YOU" if st.player_to_move else "BOT",
           "seed": game["seed"],
           "you": _side_dict(your, your.actions if st.player_to_move else None,
                             reveal_private=True),
           "bot": _side_dict(bot, bot.actions if not st.player_to_move else None),
           "hint": hint,
           "hist": (game.get("history") or [])[-3:]}
    return obj


def print_json(obj):
    print(json.dumps(obj, separators=(",", ":")))


def make_game(seed, depth, temp, game_index=None, force_ai_first=None,
              force_human_first=None):
    rng = random.Random(seed)
    types = list(Type)
    state = initial(tuple(rng.choice(types) for _ in range(3)),
                    tuple(rng.choice(types) for _ in range(3)), rng=rng)
    if force_ai_first:
        ai_starts = True
    elif force_human_first:
        ai_starts = False
    else:
        # Random first mover: the coin is seeded by the game seed so the
        # same seed reproduces the same match.
        ai_starts = random.Random(seed * 7919 + 13).random() < 0.5
    if ai_starts:
        state = replace(state, player_to_move=False).prepare()
    bot_rng = random.Random(seed * 1000003 + 17)
    planner = Planner(depth=depth, temperature=temp, rng=bot_rng)
    return {"state": state, "planner": planner, "seed": seed,
            "ai_starts": ai_starts, "log": [], "history": [],
            "sBot": 0, "compact": False, "game_index": game_index}


def record_result(game, winner):
    """Timestamped result -> session_log.json; w/l/d -> winrate_log.json."""
    entry = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "game": game.get("game_index"),
        "seed": game["seed"],
        "seat": "ai_first" if game["ai_starts"] else "human_first",
        "winner": winner,
        "plies": len(game.get("log", [])),
    }
    sess = _read_json(_session_log_path(), [])
    sess.append(entry)
    _write_json(_session_log_path(), sess)
    wr = _read_json(_winrate_log_path(), [])
    wr.append({"seat": entry["seat"],
               "result": {"YOU": "w", "BOT": "l", "DRAW": "d"}[winner]})
    _write_json(_winrate_log_path(), wr)


def _winner_of(game):
    s = game["state"]
    if s.opponent.lost:
        return "YOU"
    if s.player.lost:
        return "BOT"
    return "DRAW"


def _finish_game_if_needed(game):
    """If the game is over, record it and either start the next game or end."""
    if not (game["state"].player.lost or game["state"].opponent.lost):
        return
    winner = _winner_of(game)
    sess = game.get("session")
    if sess is None:
        if game.get("compact"):
            print_json({"done": winner,
                        "plies": len(game.get("log", []))})
        else:
            print(f"=== {winner} WIN ===")
        return
    record_result(game, winner)
    sess["done"] += 1
    done, total = sess["done"], sess["total"]
    print(f"=== GAME {done}/{total} finished: {winner} "
          f"(seed {game['seed']}, {len(game['log'])} plies) — recorded ===")
    if done >= total:
        print("SESSION COMPLETE — all games recorded:")
        show_stats(RUN)
        if os.path.exists(_state_path()):
            os.remove(_state_path())
        return
    nxt = make_game(sess["start_seed"] + done, sess["depth"], sess["temp"],
                    game_index=done + 1)
    nxt["session"] = sess
    save(nxt)
    print(f"\nGAME {done+1}/{total} (seed {nxt['seed']}) — "
          f"first mover: {'BOT' if nxt['ai_starts'] else 'YOU'}")
    print(render(nxt["state"], done + 1, total))


def cmd_new(args):
    game = make_game(int(args.seed) if args.seed else 0, args.depth, args.temp,
                     force_ai_first=args.ai_first,
                     force_human_first=args.human_first)
    game["compact"] = args.compact
    if args.compact:
        # Same rationale as in cmd_move: never hand back a state the player
        # cannot act on. If the bot moves first, resolve its turn now so the
        # opening response already contains our budget.
        state = game["state"]
        while not (state.player_to_move
                   or state.player.lost or state.opponent.lost):
            state = run_bot(state, game["planner"], verbose=False,
                            history=game.get("history"))
            game["state"] = state
    save(game)
    if args.compact:
        print_json({"seed": game["seed"],
                    "first": "BOT" if game["ai_starts"] else "YOU",
                    "state": compact_render(game)})
    else:
        print(f"[first mover: {'BOT' if game['ai_starts'] else 'YOU'}]")
        print(render(game["state"]))


def _play_one_game(game):
    """Play one game interactively (a/s/b/sw prompts), returns the winner."""
    state, planner = game["state"], game["planner"]
    total = game.get("session", {}).get("total")
    print(f"\nGAME {game['game_index']}/{total} (seed {game['seed']}) — "
          f"first mover: {'BOT' if game['ai_starts'] else 'YOU'}")
    try:
        while not (state.player.lost or state.opponent.lost):
            show_state(state)
            if state.player_to_move:
                move = human_allocation(state, input_fn=input, output_fn=print)
                before = state
                state = apply(state, move)
                planner.observe(move.attacks, move.bonuses, move.switch,
                                budget=before.player.actions)
                show_resolution(before, move, state, "You")
            else:
                planning = GameState(state.opponent, state.player,
                                     state.turn, True)
                move = planner.choose(planning)
                before = state
                if move.attacks:
                    planner.observe_shields(before.player.shields)
                state = apply(state, move)
                show_resolution(before, move, state, "AI")
            game["state"] = state
            game["log"].append(move.label)
    except PlayerQuit:
        print("\nBye! Session stopped (partial game not recorded).")
        raise
    winner = _winner_of(game)
    show_outcome(state)
    return winner


def run_session(total, start_seed, depth, temp):
    """Run `total` interactive games back to back; record each result."""
    print(f"SESSION: {total} games | random first mover | temp={temp} "
          f"depth={depth} | enter a / s / b / sw per action; q quits")
    for game_index in range(1, total + 1):
        game = make_game(start_seed + game_index - 1, depth, temp,
                         game_index=game_index)
        game["session"] = {"total": total}
        winner = _play_one_game(game)
        record_result(game, winner)
        print(f"\n=== GAME {game_index}/{total} finished: {winner} "
              f"(seed {game['seed']}, {len(game['log'])} plies) — recorded ===")
    print("\nSESSION COMPLETE — all games recorded:")
    show_stats(RUN)


def cmd_session(args):
    try:
        run_session(args.games, args.seed or 0, args.depth, args.temp)
    except PlayerQuit:
        pass


def cmd_move(args):
    game = load()
    state, planner = game["state"], game["planner"]
    compact = game.get("compact", False)
    # Advance any pending bot turns first (handles random/bot first mover).
    while not state.player_to_move:
        if state.player.lost or state.opponent.lost:
            break
        state = run_bot(state, planner, verbose=not compact,
                        history=game.get("history"))
        game["state"] = state
    if state.player_to_move and args.move and args.move != "-":
        intent = parse_intent(args.move)
        a, d, b = intent[0], intent[1], intent[2]
        # The CLI switch target is 1-based; Allocation.switch_to is 0-based.
        sw = (intent[3] - 1) if intent[3] is not None else None
        # Budget ergonomics: intent, leftover is auto-banked (up to the bank
        # cap); any excess over the cap becomes shields so all actions are spent.
        budget = state.player.actions
        used = a + d + b + (1 if sw is not None else 0)
        if used > budget:
            raise SystemExit(f"budget exceeded: {used} > {budget}")
        room = MAX_BONUS - state.player.bonus
        to_bank = min(budget - used, room)
        b += to_bank
        d += (budget - used) - to_bank
        move = human_move(state, a, d, b, sw)
        before = state
        state = apply(state, move)
        planner.observe(move.attacks, move.bonuses, move.switch,
                        budget=before.player.actions)
        # The opponent's shields are revealed on EVERY resolution; remember
        # the latest revealed value as public info.
        game["sBot"] = before.opponent.shields
        if move.attacks:
            blocked = min(move.attacks, before.opponent.shields)
            landed = move.attacks - blocked
            dmg = exchange_damage(before.player.active_character
                                  if not move.switch else before.player.characters[move.switch_to],
                                  before.opponent.active_character, landed)
            hist = (f"T{before.turn} YOU: a{move.attacks} -> {dmg} dmg "
                    f"(bot held {before.opponent.shields})")
        else:
            hist = (f"T{before.turn} YOU: did not attack "
                    f"(bot held {before.opponent.shields})")
        if not compact:
            print(f"[YOU] {move.label}  (bot held {before.opponent.shields} shields)")
        game["log"].append(move.label)
        game["history"].append(hist)
    # Compact mode is for automated players, where every response costs a whole
    # model round-trip. Resolving our allocation leaves the bot to move, so a
    # naive render would hand back a state with no budget and no decision in it,
    # forcing a second "-" call that carries no information. Measured on seeds
    # 500-510 that dead call was ~49% of all round-trips. So run the bot's reply
    # here and always answer with a position the player can actually act on.
    if compact:
        while not (state.player_to_move
                   or state.player.lost or state.opponent.lost):
            state = run_bot(state, planner, verbose=False,
                            history=game.get("history"))
            game["state"] = state
    game["state"] = state
    save(game)
    if compact:
        if state.player.lost or state.opponent.lost:
            _finish_game_if_needed(game)
        else:
            print_json(compact_render(game))
    else:
        print()
        if state.player.lost or state.opponent.lost:
            _finish_game_if_needed(game)
        else:
            sess = game.get("session")
            idx, total = (sess["done"] + 1, sess["total"]) if sess else (None, None)
            print(render(state, idx, total, history=game.get("history")))


def cmd_view(args):
    game = load()
    if game.get("compact"):
        print_json(compact_render(game))
        return
    sess = game.get("session")
    idx, total = (sess["done"] + 1, sess["total"]) if sess else (None, None)
    print(render(game["state"], idx, total, history=game.get("history")))


def cmd_end(args):
    if not os.path.exists(_state_path()):
        print("no active game (state file absent)")
        return
    game = load()
    s = game["state"]
    compact = game.get("compact", False)
    if s.player.lost or s.opponent.lost:
        winner = _winner_of(game)
        record_result(game, winner)
        if compact:
            print_json({"result": winner, "seed": game["seed"],
                        "plies": len(game.get("log", []))})
        else:
            print(f"final result recorded: {winner} (seed {game['seed']})")
    else:
        if compact:
            print_json({"result": "aborted", "seed": game["seed"]})
        else:
            print(f"game aborted mid-way (seed {game['seed']}); not recorded")
    if not compact:
        print(f"human hp left {sum(c.hp for c in s.player.characters)}, "
              f"bot hp left {sum(c.hp for c in s.opponent.characters)}")
        print(f"log: {game['log']}")
    if os.path.exists(_state_path()):
        os.remove(_state_path())


def human_move(state, a, d, b, sw=None):
    # validate budget
    if sw is not None:
        if sw == state.player.active:
            sw = None
    total = a + d + b + (1 if sw is not None else 0)
    budget = state.player.actions
    if total != budget:
        raise SystemExit(f"budget mismatch: {total} != {budget} "
                         f"(use all actions; switch costs 1)")
    if b > MAX_BONUS:
        raise SystemExit("bonus above cap")
    if sw is not None and not (0 <= sw < len(state.player.characters)):
        raise SystemExit("bad switch target")
    return Allocation(a, d, b, sw)


def run_bot(state, planner, verbose=True, history=None):
    """Bot's turn: choose, resolve, and learn what it is entitled to."""
    before = state
    planning = state.__class__(state.opponent, state.player, state.turn, True)
    move = planner.choose(planning)
    after = apply(state, move)
    # The human's shields are revealed on every resolution, regardless of
    # whether the bot attacked (symmetrical with the human's UI). So the bot
    # observes them every turn.
    planner.observe_shields(before.player.shields)
    if move.attacks:
        blocked = min(move.attacks, before.player.shields)
        landed = move.attacks - blocked
        dmg = exchange_damage(planning.player.active_character
                              if not move.switch else planning.player.characters[move.switch_to],
                              before.player.active_character, landed)
        hist = (f"T{before.turn} BOT: a{move.attacks} -> {dmg} dmg "
                f"(blocked {blocked})")
        if verbose:
            print(f"[BOT] attacked {move.attacks} (blocked {blocked}, landed {landed}, dmg {dmg})")
    else:
        hist = f"T{before.turn} BOT: did not attack"
        if verbose:
            print("[BOT] did not attack")
    if history is not None:
        history.append(hist)
    return after


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    pn = sub.add_parser("new")
    pn.add_argument("--run", default="default",
                    help="per-run sandbox folder under runs/ (parallel-safe)")
    pn.add_argument("--seed", default=None)
    pn.add_argument("--ai_first", action="store_true",
                    help="bot moves first (overrides the random coin flip)")
    pn.add_argument("--human_first", action="store_true",
                    help="human moves first (overrides the random coin flip)")
    pn.add_argument("--depth", type=int, default=2)
    pn.add_argument("--temp", type=float, default=0.12)
    pn.add_argument("--compact", action="store_true",
                    help="compact JSON output (LLM-friendly, one line per turn)")
    ps = sub.add_parser("session")
    ps.add_argument("--run", default="default")
    ps.add_argument("--games", type=int, default=20)
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--depth", type=int, default=2)
    ps.add_argument("--temp", type=float, default=0.12)
    pm = sub.add_parser("move"); pm.add_argument("--run", default="default")
    pm.add_argument("move")
    pv = sub.add_parser("view"); pv.add_argument("--run", default="default")
    pe = sub.add_parser("end"); pe.add_argument("--run", default="default")
    pst = sub.add_parser("stats"); pst.add_argument("--run", default="default")
    args = p.parse_args()
    _set_run(args.run)
    if args.cmd == "new":
        cmd_new(args)
    elif args.cmd == "session":
        cmd_session(args)
    elif args.cmd == "move":
        cmd_move(args)
    elif args.cmd == "view":
        cmd_view(args)
    elif args.cmd == "end":
        cmd_end(args)
    elif args.cmd == "stats":
        show_stats(args.run)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
