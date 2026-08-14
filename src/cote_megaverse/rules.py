"""Authoritative COTE Megaverse rules and immutable game transitions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Iterable
from random import Random

BASE_HP = 6000
BASE_ATK = 2000
HP_POOL = (5700, 5800, 5900, 6000, 6100, 6200, 6300)
ATK_POOL = (1900, 2000, 2100)
MAX_BONUS = 4
MAX_ACTIONS = 8
TURN_ACTIONS = {1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4}


class Type(IntEnum):
    A = 0
    B = 1
    C = 2
    D = 3


def multiplier(attacker: Type, defender: Type) -> float:
    if (int(attacker) + 1) % 4 == int(defender):
        return 1.3
    if (int(defender) + 1) % 4 == int(attacker):
        return 0.7
    return 1.0


def rounded_damage(value: float) -> int:
    return round(value / 100) * 100


def exchange_damage(attacker: Character, defender: Character, attacks: int) -> int:
    """Round one aggregate exchange once, after multiplying all hits."""
    if attacks <= 0:
        return 0
    return rounded_damage(attacker.atk * multiplier(attacker.type, defender.type) * attacks)


def attacks_to_kill(attacker: Character, defender: Character, shields: int = 0,
                    max_attacks: int = MAX_ACTIONS) -> int | None:
    """Find the minimum total attacks, including shields, for a lethal exchange."""
    for total in range(max(0, shields) + 1, max_attacks + 1):
        if exchange_damage(attacker, defender, total - shields) >= defender.hp:
            return total
    return None


def base_budget(turn: int) -> int:
    return min(TURN_ACTIONS.get(turn, 4), 4)


@dataclass(frozen=True, slots=True)
class Character:
    type: Type
    hp: int = BASE_HP
    atk: int = BASE_ATK
    max_hp: int = BASE_HP

    @property
    def alive(self):
        return self.hp > 0


@dataclass(frozen=True, slots=True)
class Side:
    characters: tuple[Character, ...]
    active: int = 0
    stack_order: tuple[int, ...] = ()
    bonus: int = 0
    shields: int = 0
    actions: int = 0
    voluntary_switch_used: bool = False
    forced_promotion: bool = False

    @property
    def active_character(self):
        return self.characters[self.active]

    @property
    def alive_count(self):
        return sum(character.alive for character in self.characters)

    @property
    def lost(self):
        return self.alive_count == 0

    def normalized_order(self):
        return self.stack_order or tuple(range(len(self.characters)))


def next_budget(turn: int, side: Side) -> int:
    """Public budget available on the side's next turn."""
    return min(MAX_ACTIONS, base_budget(turn) + side.bonus)


@dataclass(frozen=True, slots=True)
class GameState:
    player: Side
    opponent: Side
    turn: int = 1
    player_to_move: bool = True

    def prepare(self):
        side = self.player if self.player_to_move else self.opponent
        total = min(MAX_ACTIONS, base_budget(self.turn) + side.bonus)
        spent = min(side.bonus, total - base_budget(self.turn))
        prepared = replace(side, bonus=side.bonus - spent, actions=total,
                           voluntary_switch_used=False)
        if self.player_to_move:
            return replace(self, player=prepared)
        return replace(self, opponent=prepared)


@dataclass(frozen=True, slots=True)
class Allocation:
    attacks: int
    defends: int
    bonuses: int
    switch_to: int | None = None

    @property
    def switch(self):
        return self.switch_to is not None

    @property
    def label(self):
        suffix = f" switch={self.switch_to}" if self.switch else ""
        return f"a{self.attacks}/d{self.defends}/b{self.bonuses}{suffix}"


def legal_allocations(side: Side) -> tuple[Allocation, ...]:
    """Enumerate every legal unordered allocation, including switches."""
    def allocations(budget: int):
        capacity = max(0, MAX_BONUS - side.bonus)
        for attacks in range(budget + 1):
            for defends in range(budget - attacks + 1):
                bonuses = budget - attacks - defends
                if bonuses <= capacity:
                    yield Allocation(attacks, defends, bonuses)

    result = list(allocations(side.actions))
    if side.actions and not side.voluntary_switch_used:
        for index, character in enumerate(side.characters):
            if index == side.active or not character.alive:
                continue
            result.extend(replace(move, switch_to=index)
                         for move in allocations(side.actions - 1))
    return tuple(result)


def apply(state: GameState, allocation: Allocation) -> GameState:
    """Resolve one allocation exactly and return the next prepared state."""
    actor = state.player if state.player_to_move else state.opponent
    target = state.opponent if state.player_to_move else state.player
    budget = actor.actions - (1 if allocation.switch else 0)
    if allocation.attacks + allocation.defends + allocation.bonuses != budget:
        raise ValueError("allocation must spend every remaining action")
    active = actor.active
    switch_used = actor.voluntary_switch_used
    order = list(actor.normalized_order())
    if allocation.switch:
        if allocation.switch_to == active or not actor.characters[allocation.switch_to].alive:
            raise ValueError("invalid switch target")
        active = allocation.switch_to
        switch_used = True
        order.remove(active)
        order.insert(0, active)
    blocked = min(allocation.attacks, target.shields)
    hits = allocation.attacks - blocked
    damage = exchange_damage(actor.characters[active], target.active_character, hits)
    characters = list(target.characters)
    victim = characters[target.active]
    characters[target.active] = replace(victim, hp=max(0, victim.hp - damage))
    target_active = target.active
    forced = target.forced_promotion
    if not characters[target_active].alive:
        # Promote the next living character from the TARGET's own stack order,
        # never from the actor's. Filter out the dead active and any dead or
        # duplicated entries, preserving relative order, and rebuild a clean
        # stack with the promoted character first.
        seen = set()
        living = []
        for i in target.normalized_order():
            if i != target.active and i not in seen and characters[i].alive:
                seen.add(i)
                living.append(i)
        forced = True
        if living:
            target_active = living[0]
            target_order = (target_active,) + tuple(i for i in living[1:])
        else:
            target_active = target.active
            target_order = tuple(target.normalized_order())
    else:
        target_order = tuple(target.normalized_order())
    new_actor = replace(actor, active=active, stack_order=tuple(order), bonus=min(MAX_BONUS, actor.bonus + allocation.bonuses),
                        shields=allocation.defends, actions=0,
                        voluntary_switch_used=switch_used)
    new_target = replace(target, characters=tuple(characters), active=target_active,
                         stack_order=tuple(target_order),
                         shields=0, forced_promotion=forced)
    if state.player_to_move:
        return GameState(new_actor, new_target, state.turn + 1, False).prepare()
    return GameState(new_target, new_actor, state.turn + 1, True).prepare()


def initial(player: Iterable[Type], opponent: Iterable[Type], rng: Random | None = None) -> GameState:
    rng = rng or Random()

    def make(team):
        characters = tuple(
            Character(item, hp := rng.choice(HP_POOL), rng.choice(ATK_POOL), hp)
            for item in team
        )
        return Side(characters, stack_order=tuple(range(3)))

    return GameState(make(tuple(player)), make(tuple(opponent))).prepare()
