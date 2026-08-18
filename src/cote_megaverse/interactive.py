"""Human-vs-planner terminal UI, matching the historical battle flow."""

from dataclasses import replace
from random import Random

from .agent import Planner
from .rules import (Allocation, GameState, Type, apply, attacks_to_kill,
                    base_budget, exchange_damage, initial, legal_allocations,
                    multiplier, rounded_damage)

EMOJI = {Type.A: "[A]", Type.B: "[B]", Type.C: "[C]", Type.D: "[D]"}
TYPE_NAMES = {Type.A: "Artist", Type.B: "Brawler", Type.C: "Coordinator", Type.D: "Defender"}
MAX_BONUS = 4
SEARCH_DEPTH = 3


class PlayerQuit(Exception):
    pass


def bar(value, maximum, width=12):
    filled = int(max(0, min(value / max(1, maximum), 1.0)) * width)
    return "#" * filled + "." * (width - filled)


def parse_team(value):
    team = tuple(Type[item.strip().upper()] for item in value.split(",") if item.strip())
    if len(team) != 3:
        raise ValueError("team must contain exactly three types")
    return team


def _available_moves(state):
    return legal_allocations(state.player)


def _parse_switch(raw):
    try:
        return int(raw) - 1
    except ValueError:
        return None


def parse_allocation(value, state):
    """Parse a compact allocation string for tests and scripted clients."""
    fields = [item.strip() for item in value.split(",")]
    if len(fields) not in (3, 4) or not all(item.isdigit() for item in fields):
        raise ValueError("use attacks,defends,bonuses[,switch_index]")
    values = [int(item) for item in fields]
    move = Allocation(values[0], values[1], values[2], values[3] if len(values) == 4 else None)
    if move not in legal_allocations(state.player):
        raise ValueError("illegal allocation for current budget")
    return move


def human_allocation(state, input_fn=input, output_fn=print):
    """Queue actions with the historical a/s/b/sw interaction."""
    actions = state.player.actions
    attack = defend = bonus = 0
    switch_to = None
    while attack + defend + bonus + (1 if switch_to is not None else 0) < actions:
        remaining = actions - attack - defend - bonus - (1 if switch_to is not None else 0)
        output_fn(f"\n  Actions: {remaining} | Your bonus: {state.player.bonus}/{MAX_BONUS}")
        command = input_fn("  a=atk s=shield b=bonus sw=switch > ").strip().lower()
        if command == "q":
            raise PlayerQuit
        if command == "a":
            attack += 1
        elif command == "s":
            defend += 1
        elif command == "b":
            if state.player.bonus + bonus >= MAX_BONUS:
                output_fn("  [MAX] Bonus full!")
            else:
                bonus += 1
        elif command == "sw":
            if switch_to is not None or state.player.voluntary_switch_used:
                output_fn("  Already switched!")
                continue
            output_fn("  Targets:")
            for index, character in enumerate(state.player.characters):
                if index != state.player.active and character.alive:
                    damage = exchange_damage(character, state.opponent.active_character, 1)
                    output_fn(f"    [{index + 1}] {EMOJI[character.type]} {TYPE_NAMES[character.type]:10s} "
                              f"HP:{character.hp} dmg:{damage}/hit")
            target = _parse_switch(input_fn("  Number (0=cancel): ").strip())
            if target is not None and any(move.switch_to == target for move in _available_moves(state)):
                switch_to = target
            else:
                output_fn("  Switch cancelled")
        else:
            output_fn("  Choose an action: a, s, b, or sw")
    move = Allocation(attack, defend, bonus, switch_to)
    if move not in _available_moves(state):
        raise ValueError("constructed illegal allocation")
    return move


def show_state(state, battle_number=None):
    public_player = state.player
    public_opponent = state.opponent
    if battle_number is not None:
        print(f"\n{'=' * 55}\n  BATTLE #{battle_number}\n{'=' * 55}")
    print("\n" + "=" * 55)
    print(f"  TURN #{state.turn} -- {'Your turn' if state.player_to_move else 'Bot turn'}")
    print("=" * 55)
    _show_side("YOU", public_player, hide_shields=False)
    print("  ---- AI ----")
    _show_side("AI", public_opponent, hide_shields=True)
    active_side = public_player if state.player_to_move else public_opponent
    base = base_budget(state.turn)
    bonus_actions = max(0, active_side.actions - base)
    if state.player_to_move:
        bonus = "*" * active_side.bonus + "." * (MAX_BONUS - active_side.bonus)
        print(f"  Bonus: {bonus}  (Actions: {base} + {bonus_actions})")
    else:
        print(f"  Actions: {base} + {bonus_actions}")
    if state.player.lost or state.opponent.lost:
        return
    active = public_player.active_character
    target = public_opponent.active_character
    mult = multiplier(active.type, target.type)
    damage = exchange_damage(active, target, 1)
    tag = "[OK]" if mult > 1 else "[XX]" if mult < 1 else "--"
    print(f"\n  {EMOJI[active.type]} vs {EMOJI[target.type]}: x{mult} {tag}")
    print(f"  Pot dmg: ~{damage}")
    attacks_needed = attacks_to_kill(active, target, shields=0)
    if attacks_needed is not None and attacks_needed <= public_player.actions:
        print(f"  [KILL possible] {attacks_needed} attacks needed")


def _show_side(label, side, hide_shields):
    if label == "YOU":
        print("  YOU")
    order = (side.active,) + tuple(index for index in side.stack_order
                                   if index != side.active and side.characters[index].alive)
    for index in order:
        character = side.characters[index]
        marker = ">" if index == side.active else " "
        dead = " [X]" if not character.alive else ""
        shields = f" [S]x{side.shields}" if not hide_shields and index == side.active and side.shields else ""
        print(f"  {marker} {EMOJI[character.type]}#{index + 1} {TYPE_NAMES[character.type]:10s} "
              f"|{bar(character.hp, character.max_hp)}| {character.hp:4d} ATK:{character.atk}{shields}{dead}")


def show_resolution(before, move, after, label):
    defender = before.opponent if before.player_to_move else before.player
    attacker = before.player if before.player_to_move else before.opponent
    blocked = min(move.attacks, defender.shields)
    landed = max(0, move.attacks - blocked)
    # If the move included a switch, the attacks come from the switched-to
    # character, not the pre-switch active. Report the real attacker.
    attacker_character = (attacker.characters[move.switch_to]
                          if move.switch else attacker.active_character)
    target = defender.active_character
    damage = min(target.hp, exchange_damage(attacker_character, target, landed))
    print("\n" + "-" * 40)
    print(f"  {label} turn #{before.turn}")
    print("-" * 40)
    if move.switch:
        print(f"  {label} switched to #{move.switch_to + 1}")
    if label == "You":
        print(f"  AI spent {defender.shields} shields")
    if move.attacks:
        if blocked == move.attacks:
            print("  BLOCK!")
            print(f"  {label}: {move.attacks} attacks vs {defender.shields} shields")
        else:
            print(f"  {label}: {move.attacks} attacks vs {defender.shields} shields, "
                  f"{landed} hit, {damage} dmg")
    elif move.defends or move.bonuses:
        print(f"  {label} did not attack")


def show_outcome(state, output_fn=print):
    winner = "AI" if state.player.lost else "YOU" if state.opponent.lost else "DRAW"
    result = "YOU LOSE" if winner == "AI" else "YOU WIN" if winner == "YOU" else "DRAW"
    output_fn(f"\n  {result}")
    return winner


def play(player_team, bot_team, depth=SEARCH_DEPTH, input_fn=input, output_fn=print, max_turns=100):
    state = initial(player_team, bot_team, rng=Random())
    bot = Planner(depth=depth)
    while state.turn <= max_turns and not state.player.lost and not state.opponent.lost:
        show_state(state)
        if state.player_to_move:
            move = human_allocation(state, input_fn, output_fn)
            before = state
            state = apply(state, move)
            bot.observe(move.attacks, move.bonuses, move.switch,
                        budget=before.player.actions, turn=before.turn)
            show_resolution(before, move, state, "You")
        else:
            planning = GameState(state.opponent, state.player, state.turn, True)
            move = bot.choose(planning)
            before = state
            # Human's shields are revealed every resolution, attack or not.
            bot.observe_shields(before.player.shields)
            state = apply(state, move)
            show_resolution(before, move, state, "AI")
    winner = show_outcome(state, output_fn)
    return winner, state


def main():
    print("\n" + "=" * 55)
    print("  COTE MEGAVERSE")
    print("=" * 55)
    try:
        rng = Random()
        teams = [tuple(rng.choice(tuple(Type)) for _ in range(3)) for _ in range(2)]
        play(teams[0], teams[1])
    except PlayerQuit:
        print("\nBye!")


if __name__ == "__main__":
    main()
