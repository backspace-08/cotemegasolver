"""Authoritative COTE Megaverse rules and immutable game transitions."""

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Iterable

BASE_HP = 6000
BASE_ATK = 2000
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


def base_budget(turn: int) -> int:
    return min(TURN_ACTIONS.get(turn, 4), 4)


@dataclass(frozen=True)
class Character:
    type: Type
    hp: int = BASE_HP
    atk: int = BASE_ATK

    @property
    def alive(self):
        return self.hp > 0


@dataclass(frozen=True)
class Side:
    characters: tuple[Character, ...]
    active: int = 0
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


def next_budget(turn: int, side: Side) -> int:
    """Public budget available on the side's next turn."""
    return min(MAX_ACTIONS, base_budget(turn) + side.bonus)


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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
    if allocation.switch:
        if allocation.switch_to == active or not actor.characters[allocation.switch_to].alive:
            raise ValueError("invalid switch target")
        active = allocation.switch_to
        switch_used = True
    blocked = min(allocation.attacks, target.shields)
    hits = allocation.attacks - blocked
    damage = rounded_damage(
        actor.characters[active].atk * multiplier(
            actor.characters[active].type, target.active_character.type) * hits)
    characters = list(target.characters)
    victim = characters[target.active]
    characters[target.active] = replace(victim, hp=max(0, victim.hp - damage))
    target_active = target.active
    forced = target.forced_promotion
    if not characters[target_active].alive:
        alive = [i for i, character in enumerate(characters) if character.alive]
        target_active = alive[0] if alive else target_active
        forced = True
    new_actor = replace(actor, active=active, bonus=min(MAX_BONUS, actor.bonus + allocation.bonuses),
                        shields=allocation.defends, actions=0,
                        voluntary_switch_used=switch_used)
    new_target = replace(target, characters=tuple(characters), active=target_active,
                         shields=0, forced_promotion=forced)
    if state.player_to_move:
        return replace(GameState(new_actor, new_target, state.turn + 1, False)).prepare()
    return replace(GameState(new_target, new_actor, state.turn + 1, True)).prepare()


def initial(player: Iterable[Type], opponent: Iterable[Type]) -> GameState:
    make = lambda team: Side(tuple(Character(item) for item in team))
    return GameState(make(tuple(player)), make(tuple(opponent))).prepare()
