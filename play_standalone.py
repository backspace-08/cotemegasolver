"""Self-contained COTE Megaverse harness: human-vs-Planner in a single file.

This is play.py with every dependency (rules, opponent infoset, strategy,
Planner, terminal UI) inlined into one module. It runs on its own in any
environment with just this file and Python >= 3.10 — no `src` package needed.

Fair observation only: the human sees HP, ATK, types, actives, stacks, turn,
the acting side's public budget, and public resolutions. The bot's hidden bank
and held shields are never shown; the planner derives them via OpponentModel
belief, never by reading resolver secrets.

Usage (identical to play.py):
  python play_standalone.py new --seed N [--ai_first|--human_first] [--temp T]
  python play_standalone.py move "a,d,b" [sw]
  python play_standalone.py view
  python play_standalone.py end
  python play_standalone.py session --games 20 [--seed S] [--temp T]
      Session mode: play N games in a row; after each game the result is
      recorded (with timestamp) to session_log.json and winrate_log.json,
      then the next game starts immediately. You just play.
"""
import argparse
import datetime
import json
import math
import os
import pickle
import random
import re
import sys

from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Iterable
from random import Random

# =========================================================================
# rules.py
# =========================================================================
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


@dataclass(frozen=True)
class Character:
    type: Type
    hp: int = BASE_HP
    atk: int = BASE_ATK
    max_hp: int = BASE_HP

    @property
    def alive(self):
        return self.hp > 0


@dataclass(frozen=True)
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
        # never from the actor's.
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
        return replace(GameState(new_actor, new_target, state.turn + 1, False)).prepare()
    return replace(GameState(new_target, new_actor, state.turn + 1, True)).prepare()


def initial(player: Iterable[Type], opponent: Iterable[Type], rng: Random | None = None) -> GameState:
    rng = rng or Random()

    def make(team):
        characters = tuple(
            Character(item, hp := rng.choice(HP_POOL), rng.choice(ATK_POOL), hp)
            for item in team
        )
        return Side(characters, stack_order=tuple(range(3)))

    return GameState(make(tuple(player)), make(tuple(opponent))).prepare()


# =========================================================================
# infoset.py
# =========================================================================
def _binomial(n: int, k: int) -> float:
    """Small binomial coefficient for split weighting."""
    if k < 0 or k > n:
        return 0.0
    result = 1.0
    for i in range(min(k, n - k)):
        result = result * (n - i) / (i + 1)
    return result


def legal_splits(remainder: int, capacity: int = MAX_BONUS):
    """Every legal ``(shields, bank)`` split of a public remainder."""
    if remainder <= 0:
        return ((0, 0),)
    return tuple((remainder - bank, bank)
                 for bank in range(min(capacity, remainder) + 1))


@dataclass(frozen=True)
class World:
    """One consistent hypothesis about the opponent's hidden state."""

    shields: int
    bank: int
    probability: float


@dataclass(frozen=True)
class TurnRecord:
    """One completed opponent turn as seen from public information."""

    turn: int
    budget: int
    attacks: int
    switched: bool
    remainder: int
    revealed_bank: int
    confirmed_shields: int | None = None


@dataclass
class OpponentModel:
    """Public-information belief over an opponent's hidden shields and bank.

    The model is a small weighted set of ``(shields, bank)`` candidates. It is
    narrowed by evidence, never by assumption.
    """

    records: list[TurnRecord] = field(default_factory=list)
    _candidates: dict[tuple[int, int], float] = field(
        default_factory=lambda: {(0, 0): 1.0})
    _defend_count: float = 1.0
    _bank_count: float = 1.0
    _attack_actions: float = 0.0
    _total_actions: float = 0.0

    # ---------------------------------------------------------------- observe

    def observe_turn(self, turn: int, budget: int, attacks: int,
                     switched: bool = False):
        revealed_bank = max(0, budget - base_budget(turn))
        self._confirm_previous_bank(revealed_bank)
        remainder = max(0, budget - attacks - int(switched))
        self.records.append(TurnRecord(
            turn=turn, budget=budget, attacks=attacks, switched=switched,
            remainder=remainder, revealed_bank=revealed_bank))
        self._attack_actions += attacks
        self._total_actions += max(1, budget)
        self._candidates = self._prior(remainder)

    def observe_our_attack(self, attacks: int, blocked: int):
        """Record what our own attack learned about their shields."""
        if attacks <= 0:
            return
        if blocked < attacks:
            self._restrict(lambda shields, bank: shields == blocked)
        else:
            self._restrict(lambda shields, bank: shields >= attacks)
        total = sum(self._candidates.values()) or 1.0
        exp_shields = sum(shields * w for (shields, _), w in self._candidates.items()) / total
        exp_bank = sum(bank * w for (_, bank), w in self._candidates.items()) / total
        self._defend_count += exp_shields
        self._bank_count += exp_bank

    def expire_shields(self):
        """Clear held shields after our allocation resolved. Bank survives."""
        collapsed: dict[tuple[int, int], float] = {}
        for (_, bank), weight in self._candidates.items():
            collapsed[(0, bank)] = collapsed.get((0, bank), 0.0) + weight
        self._candidates = collapsed or {(0, 0): 1.0}

    # ----------------------------------------------------------------- belief

    def worlds(self) -> tuple[World, ...]:
        """Live joint hypotheses, most likely first."""
        total = sum(self._candidates.values()) or 1.0
        ordered = sorted(self._candidates.items(),
                         key=lambda item: (-item[1], item[0]))
        return tuple(World(shields, bank, weight / total)
                     for (shields, bank), weight in ordered)

    def shield_distribution(self) -> dict[int, float]:
        """Marginal over currently held shields."""
        result: dict[int, float] = {}
        for world in self.worlds():
            result[world.shields] = result.get(world.shields, 0.0) + world.probability
        return result

    def bank_distribution(self) -> dict[int, float]:
        """Marginal over the bank they carry into their next turn."""
        result: dict[int, float] = {}
        for world in self.worlds():
            result[world.bank] = result.get(world.bank, 0.0) + world.probability
        return result

    def next_budget_distribution(self, opponent_turn: int) -> dict[int, float]:
        """Distribution over the opponent's next public action budget."""
        base = base_budget(opponent_turn)
        result: dict[int, float] = {}
        for bank, probability in self.bank_distribution().items():
            budget = min(MAX_ACTIONS, base + bank)
            result[budget] = result.get(budget, 0.0) + probability
        return result

    @property
    def exact(self) -> bool:
        """True when only one world survives, so no bluff space remains."""
        return len(self._candidates) == 1

    @property
    def max_shields(self) -> int:
        return max(shields for shields, _ in self._candidates)

    @property
    def attack_rate(self) -> float:
        if self._total_actions <= 0:
            return 1 / 3
        return self._attack_actions / self._total_actions

    @property
    def defend_share(self) -> float:
        """Behavioural estimate of how they split remainders, not a secret."""
        total = self._defend_count + self._bank_count
        return self._defend_count / total if total else 0.5

    # ---------------------------------------------------------------- helpers

    def _prior(self, remainder: int) -> dict[tuple[int, int], float]:
        """Weight each legal split by observed splitting behaviour."""
        EPSILON = 0.02
        share = min(0.98, max(0.02, self.defend_share))
        weights: dict[tuple[int, int], float] = {}
        for shields, bank in legal_splits(remainder):
            weight = (_binomial(shields + bank, shields)
                      * share ** shields
                      * (1.0 - share) ** bank)
            weights[(shields, bank)] = weight + EPSILON
        return weights

    def _restrict(self, predicate):
        """Keep only worlds consistent with new evidence."""
        surviving = {key: weight for key, weight in self._candidates.items()
                     if predicate(*key)}
        if surviving:
            self._candidates = surviving

    def _confirm_previous_bank(self, revealed_bank: int):
        """A revealed budget proves the bank, hence the earlier split."""
        if not self.records:
            return
        self._restrict(lambda shields, bank: bank == revealed_bank)
        surviving = [key for key in self._candidates if key[1] == revealed_bank]
        if len(surviving) == 1:
            shields = surviving[0][0]
            self.records[-1] = TurnRecord(
                **{**self.records[-1].__dict__, "confirmed_shields": shields})
            self._defend_count += shields
            self._bank_count += revealed_bank


# =========================================================================
# strategy.py
# =========================================================================
@dataclass(frozen=True)
class SwitchValue:
    target: int
    current_damage: int
    target_damage: int
    current_received: int
    target_received: int
    action_cost: int
    value: int
    recommended: bool


def marginal_bonus_value(state: GameState, bonus: int) -> int:
    """Value of one stored action by comparing next-turn pressure."""
    side = state.player
    target = state.opponent
    current_budget = side.actions or next_budget(state.turn, side)
    future_budget = min(8, current_budget + max(0, bonus))
    current_damage = exchange_damage(
        side.active_character, target.active_character,
        max(0, current_budget - target.shields))
    future_damage = exchange_damage(
        side.active_character, target.active_character,
        max(0, future_budget - target.shields))
    return future_damage - current_damage


def switch_value(state: GameState, target_index: int, belief) -> SwitchValue:
    me, enemy = state.player, state.opponent
    current = me.active_character
    target = me.characters[target_index]
    enemy_character = enemy.active_character
    current_damage = exchange_damage(current, enemy_character, 1)
    target_damage = exchange_damage(target, enemy_character, 1)
    enemy_budget = next_budget(state.turn + 1, enemy)
    current_received = exchange_damage(enemy_character, current, enemy_budget)
    target_received = exchange_damage(enemy_character, target, enemy_budget)
    current_survival = max(0, current.hp - current_received)
    target_survival = max(0, target.hp - target_received)
    body_preservation = 1800 if current_survival == 0 < target_survival else 0
    reserve = (target_survival - current_survival) * 0.35
    value = int(target_damage - current_damage + reserve + body_preservation - 200)
    return SwitchValue(
        target_index, current_damage, target_damage, current_received,
        target_received, 1, value, value > 0 and state.player.actions > 1,
    )


@dataclass
class Objective:
    name: str = "normal"
    reason: str = ""
    created_turn: int = 0
    confidence: float = 0.0
    success_condition: str = ""
    abort_condition: str = ""

    def update(self, state: GameState, *, lethal_probability: float,
               expected_incoming: int, attack_rate: float,
               passive_streak: int = 0, turn: int,
               opponent_bank: float | None = None):
        if opponent_bank is None:
            opponent_bank = state.opponent.bonus
        if lethal_probability >= 1.0:
            self.name = "finish"
            self.reason = "guaranteed lethal is available"
            self.success_condition = "active opponent dies"
            self.abort_condition = "position changes before resolution"
        elif state.player.active_character.hp <= expected_incoming:
            self.name = "survive"
            self.reason = "active character is exposed to lethal threat"
            self.success_condition = "survive next exchange"
            self.abort_condition = "threat estimate falls"
        elif opponent_bank >= 3:
            self.name = "deny_burst"
            self.reason = "belief says the opponent likely banked a burst"
            self.success_condition = "survive the opponent's burst turn"
            self.abort_condition = "opponent bonus is spent or lethal appears"
        elif passive_streak >= 2:
            self.name = "break_stall"
            self.reason = "opponent has repeated non-attacking allocations"
            self.success_condition = "create damage or a lethal threat"
            self.abort_condition = "opponent resumes pressure"
        elif attack_rate > 0.60 and state.player.bonus < 3:
            self.name = "prepare_burst"
            self.reason = "opponent attacks frequently"
            self.success_condition = "bonus bank reaches three"
            self.abort_condition = "active HP becomes unsafe or lethal appears"
        else:
            self.name = "normal"
            self.reason = "no dominant tactical objective"
        self.created_turn = self.created_turn or turn
        self.confidence = max(lethal_probability, attack_rate)


# =========================================================================
# agent.py
# =========================================================================
@dataclass
class PublicHistory:
    actions: list[tuple[int, int, int, int]] = field(default_factory=list)
    events: list["PublicEvent"] = field(default_factory=list)
    held_shields: int | None = None
    policy_scores: dict[str, float] = field(default_factory=lambda: {
        "aggressive": 1.0,
        "defensive": 1.0,
        "builder": 1.0,
        "opportunistic": 1.0,
    })

    def observe(self, attacks: int, bonuses: int, switched: bool,
                budget: int | None = None):
        shields = None if budget is None else max(
            0, budget - attacks - bonuses - int(switched))
        self.actions.append((attacks, shields or 0, bonuses, int(switched)))
        self.held_shields = shields
        self.events.append(PublicEvent(attacks, bonuses, int(switched), shields))
        self._update_policy(attacks, shields or 0, bonuses)

    def observe_resolved(self, attacks, defends, bonuses, switched=False):
        self.actions.append((attacks, defends, bonuses, int(switched)))
        self.held_shields = defends
        self.events.append(PublicEvent(attacks, bonuses, int(switched), defends))
        self._update_policy(attacks, defends, bonuses)

    def reveal_latest_defends(self, defends: int):
        if not self.actions:
            return
        attacks, _, bonuses, switched = self.actions[-1]
        self.actions[-1] = (attacks, defends, bonuses, switched)
        self.held_shields = defends
        if self.events:
            event = self.events[-1]
            self.events[-1] = replace(event, resolved_shields=defends)
        self._update_policy(attacks, defends, bonuses)

    def _update_policy(self, attacks: int, defends: int, bonuses: int):
        total = max(1, attacks + defends + bonuses)
        shares = {
            "aggressive": attacks / total,
            "defensive": defends / total,
            "builder": bonuses / total,
            "opportunistic": max(attacks, bonuses) / total,
        }
        for name, share in shares.items():
            self.policy_scores[name] = self.policy_scores[name] * 0.8 + share

    @property
    def policy_belief(self):
        total = sum(self.policy_scores.values()) or 1.0
        return {name: score / total for name, score in self.policy_scores.items()}

    @property
    def attack_rate(self):
        total = sum(a + d + b for a, d, b, _ in self.actions)
        return sum(a for a, _, _, _ in self.actions) / total if total else 1 / 3

    @property
    def defend_rate(self):
        total = sum(a + d + b for a, d, b, _ in self.actions)
        return sum(d for _, d, _, _ in self.actions) / total if total else 1 / 3

    @property
    def bonus_rate(self):
        total = sum(a + d + b for a, d, b, _ in self.actions)
        return sum(b for _, _, b, _ in self.actions) / total if total else 1 / 3

    @property
    def passive_streak(self):
        streak = 0
        for attacks, _, _, _ in reversed(self.actions):
            if attacks:
                break
            streak += 1
        return streak


@dataclass(frozen=True)
class PublicEvent:
    """Facts visible after an allocation, never hidden resolver intent."""

    attacks: int
    bonuses: int
    switched: int
    resolved_shields: int | None = None


@dataclass(frozen=True)
class ShieldBelief:
    probabilities: dict[int, float]

    @property
    def expected(self):
        return sum(k * v for k, v in self.probabilities.items())

    @property
    def maximum(self):
        return max(self.probabilities)


@dataclass(frozen=True)
class AllocationHypothesis:
    allocation: Allocation
    probability: float


@dataclass(frozen=True)
class TacticalFacts:
    damage_per_hit: int
    hits_to_kill: int
    attacks_for_guaranteed_lethal: int
    lethal_probability: float
    guaranteed_lethal: bool
    expected_damage: float


@dataclass(frozen=True)
class TacticalOutcome:
    guaranteed_lethal: bool
    kill_and_defend: bool
    guaranteed_immediate_loss: bool
    lethal_probability: float
    wins_match: bool = False
    loss_probability: float = 0.0


class Planner:
    def __init__(self, depth=3, branch_limit=None, temperature=0.0, rng=None,
                 band_fraction=0.15):
        self.depth = depth
        self.branch_limit = branch_limit
        # Context-conditional mixing. 0 -> fully deterministic argmax (fast,
        # reproducible, strong, good for tests). > 0 -> the bot mixes ONLY at
        # genuine decision points: when at least two moves score within
        # `band_fraction` of the best, it samples among them with a softmax of
        # width `temperature` (fraction of the best score).
        self.temperature = temperature
        self.band_fraction = band_fraction
        self.rng = rng
        self.history = PublicHistory()
        self.model = OpponentModel()
        self.objective = Objective()
        self.last_report = {}
        self._search_cache = {}
        self.passive_streak = 0
        self._observed_turns = 0
        self._believed_bank = 0

    def observe(self, attacks, bonuses=None, switched=False, budget=None,
                turn=None):
        """Record a public opponent allocation."""
        if turn is None:
            turn = self._observed_turns * 2 + 1
        self._observed_turns += 1
        if budget is None:
            budget = attacks + int(switched)
        self.model.observe_turn(turn, budget, attacks, switched)
        self.history.observe(attacks, bonuses or 0, switched, budget)

    def observe_shields(self, shields: int):
        """Our attack met this many shields, which pins the live worlds."""
        self.model.observe_our_attack(shields + 1, shields)
        self.history.reveal_latest_defends(shields)

    def belief(self, opponent: GameState | object) -> ShieldBelief:
        """Distribution over the opponent's currently held shields."""
        return ShieldBelief(self.model.shield_distribution())

    def opponent_budgets(self, state: GameState) -> dict[int, float]:
        """Distribution over the opponent's next public budget."""
        budget_turn = state.turn + (1 if state.player_to_move else 0)
        distribution = self.model.next_budget_distribution(budget_turn)
        return distribution or {base_budget(budget_turn): 1.0}

    def opponent_allocations(self, state: GameState) -> tuple[AllocationHypothesis, ...]:
        """Estimate legal next allocations using only public behavior history."""
        side = state.opponent
        hypotheses: list[AllocationHypothesis] = []
        for budget, budget_probability in self.opponent_budgets(state).items():
            forecast = replace(side, bonus=0, actions=budget)
            candidates = legal_allocations(forecast)
            weights = []
            for move in candidates:
                total = max(1, move.attacks + move.defends + move.bonuses)
                attack_share = move.attacks / total
                defend_share = move.defends / total
                bonus_share = move.bonuses / total
                weight = (
                    0.15
                    + attack_share * self.history.attack_rate
                    + defend_share * self.history.defend_rate
                    + bonus_share * self.history.bonus_rate
                )
                policy = self.history.policy_belief
                weight += policy["aggressive"] * attack_share * 0.2
                weight += policy["defensive"] * defend_share * 0.2
                weight += policy["builder"] * bonus_share * 0.2
                weights.append(weight)
            total_weight = sum(weights) or 1.0
            hypotheses.extend(
                AllocationHypothesis(move, budget_probability * weight / total_weight)
                for move, weight in zip(candidates, weights)
            )
        return tuple(hypotheses)

    def facts(self, state: GameState, allocation: Allocation, belief: ShieldBelief):
        me, enemy = state.player, state.opponent
        character = me.characters[allocation.switch_to if allocation.switch else me.active]
        target = enemy.active_character
        damage = exchange_damage(character, target, 1)
        needed = attacks_to_kill(character, target, 0) or MAX_ACTIONS + 1
        lethal = sum(p for shields, p in belief.probabilities.items()
                     if attacks_to_kill(character, target, shields) is not None
                     and allocation.attacks >= attacks_to_kill(character, target, shields))
        expected = sum(min(target.hp, exchange_damage(
            character, target, max(0, allocation.attacks - shields))) * p
                       for shields, p in belief.probabilities.items())
        return TacticalFacts(
            damage, needed, needed + belief.maximum, lethal,
            all(allocation.attacks >= (attacks_to_kill(character, target, shields) or MAX_ACTIONS + 1)
                 for shields in belief.probabilities), expected)

    def _burst_setup_value(self, state, move, belief, attacker_idx):
        """Value of banking `move.bonuses` now to kill on the bot's next turn."""
        bank_after = move.bonuses
        if bank_after <= 0:
            return 0.0
        me, enemy = state.player, state.opponent
        attacker = me.characters[attacker_idx]
        target = enemy.active_character
        if attacker_idx == me.active:
            opp_turn = state.turn + 1
            worst_bank = max(self.model.bank_distribution().keys(), default=0)
            opp_budget = min(MAX_ACTIONS, base_budget(opp_turn) + worst_bank)
            incoming = exchange_damage(enemy.active_character, attacker,
                                       max(0, opp_budget - move.defends))
            if incoming >= attacker.hp:
                return 0.0
        my_next_turn = state.turn + 2
        next_budget_ = min(MAX_ACTIONS, base_budget(my_next_turn) + bank_after)
        for shields, _p in belief.probabilities.items():
            landed = max(0, next_budget_ - shields)
            if exchange_damage(attacker, target, landed) >= target.hp:
                return 5000.0 * bank_after
        return 0.0

    def _reply_kills_us(self, child: GameState, bank: int) -> bool:
        """Can the opponent's next allocation end the match against us?"""
        if child.player.alive_count > 1:
            return False
        defender = child.player.active_character
        budget = min(MAX_ACTIONS, base_budget(child.turn) + max(0, bank))
        attacker_side = child.opponent
        for index, character in enumerate(attacker_side.characters):
            if not character.alive:
                continue
            switch_cost = 0 if index == attacker_side.active else 1
            attacks = max(0, budget - switch_cost - child.player.shields)
            if exchange_damage(character, defender, attacks) >= defender.hp:
                return True
        return False

    def tactical_outcome(self, state: GameState, move: Allocation,
                         belief: ShieldBelief, facts: TacticalFacts):
        """Evaluate hard tactical facts across every live joint world."""
        guaranteed_loss = True
        wins_match = True
        loss_weight = 0.0
        for world in self.model.worlds():
            resolver_state = replace(state, opponent=replace(
                state.opponent, shields=world.shields, bonus=0))
            child = apply(resolver_state, move)
            if not child.opponent.lost:
                wins_match = False
            world_loss = (not child.opponent.lost
                          and self._reply_kills_us(child, world.bank))
            if world_loss:
                loss_weight += world.probability
            else:
                guaranteed_loss = False
        return TacticalOutcome(
            guaranteed_lethal=facts.guaranteed_lethal,
            kill_and_defend=facts.guaranteed_lethal and move.defends > 0,
            guaranteed_immediate_loss=guaranteed_loss,
            lethal_probability=facts.lethal_probability,
            wins_match=wins_match,
            loss_probability=loss_weight,
        )

    def choose(self, state: GameState) -> Allocation:
        # Resolver states contain secrets. Enforce the public-information
        # boundary here so callers cannot accidentally expose the enemy's held
        # shields or stored bank.
        state = replace(state, opponent=replace(state.opponent, shields=0,
                                                bonus=0))
        bank_belief = self.model.bank_distribution()
        credible_bank = max((bank for bank, p in bank_belief.items() if p >= 0.15),
                            default=0)
        self._believed_bank = credible_bank
        self._search_cache.clear()
        effective_depth = self.depth + (
            1 if state.player.alive_count + state.opponent.alive_count <= 3 else 0)
        belief = self.belief(state)
        switch_values = [switch_value(state, index, belief)
                         for index, character in enumerate(state.player.characters)
                         if index != state.player.active and character.alive]
        candidates = legal_allocations(state.player)
        if self.branch_limit:
            candidates = tuple(sorted(candidates, key=lambda x: (x.attacks, x.bonuses, x.defends), reverse=True)[:self.branch_limit])
        candidate_facts = [(move, self.facts(state, move, belief)) for move in candidates]
        best_lethal = max((facts.lethal_probability for _, facts in candidate_facts), default=0.0)
        expected_incoming = self._expected_incoming(state, 0)
        self.objective.update(
            state,
            lethal_probability=best_lethal,
            expected_incoming=expected_incoming,
            attack_rate=self.history.attack_rate,
            passive_streak=self.passive_streak,
            turn=state.turn,
            opponent_bank=credible_bank,
        )
        switch_by_target = {value.target: value for value in switch_values}

        opponent_exposed = belief.probabilities.get(0, 0.0) >= 0.6
        opponent_banking = credible_bank >= 2
        behind = (state.player.alive_count < state.opponent.alive_count)
        scored = []
        for move, facts in candidate_facts:
            continuations = []
            for shields, probability in belief.probabilities.items():
                resolver_state = replace(
                    state, opponent=replace(state.opponent, shields=shields))
                child = apply(resolver_state, move)
                continuations.append(
                    (self._search(child, effective_depth - 1,
                                  state.player_to_move), probability))
            expected_continuation = sum(value * probability
                                        for value, probability in continuations)
            worst_continuation = min(value for value, _ in continuations)
            continuation = expected_continuation * 0.8 + worst_continuation * 0.2
            incoming = self._expected_incoming(state, move.defends)
            survival = max(0.0, state.player.active_character.hp - incoming)
            future_bonus = marginal_bonus_value(state, move.bonuses) * 0.35
            switch = switch_by_target.get(move.switch_to)
            switch_score = switch.value * 0.8 if switch else 0.0
            objective_score = self._objective_score(
                move, facts, survival, state.player.active_character.hp)
            attacker_idx = move.switch_to if move.switch else state.player.active
            burst_setup = self._burst_setup_value(state, move, belief, attacker_idx)
            score = continuation + facts.expected_damage * 0.6
            score += facts.lethal_probability * 5000 + future_bonus
            score += survival * 0.25 + switch_score + objective_score + burst_setup
            known_unshielded = belief.probabilities == {0: 1.0}
            if self.history.attack_rate >= 0.65 and known_unshielded:
                score += facts.expected_damage * 0.9 + move.attacks * 250
                if move.attacks == 0:
                    score -= 1800
            if state.turn <= 4 and known_unshielded and move.attacks:
                score += facts.expected_damage * 1.1 + move.attacks * 600
            elif state.turn <= 4 and known_unshielded and move.attacks == 0:
                score -= 2200
            if self.passive_streak >= 2 and move.attacks == 0:
                score -= 4000 + self.passive_streak * 1000
            punish = 0.0
            if opponent_exposed and opponent_banking and move.attacks:
                punish = facts.expected_damage * 0.7 + move.attacks * 300
            desperation = 0.0
            if behind:
                if move.attacks:
                    desperation = facts.expected_damage * 0.4 + move.attacks * 150
                elif move.bonuses:
                    desperation = move.bonuses * 250.0
                if move.attacks == 0 and move.defends > 0 and move.bonuses == 0:
                    desperation -= move.defends * 400.0
            deny_burst = 0.0
            if opponent_banking:
                worst_burst = min(MAX_ACTIONS, base_budget(state.turn + 1)
                                  + credible_bank)
                incoming = exchange_damage(
                    state.opponent.active_character,
                    state.player.active_character,
                    max(0, worst_burst - move.defends))
                if move.defends:
                    if incoming < state.player.active_character.hp:
                        deny_burst = 450.0 * move.defends
                    else:
                        deny_burst = -450.0 * move.defends
            score += punish + desperation + deny_burst
            components = {
                "continuation": continuation,
                "expected_damage": facts.expected_damage * 0.6,
                "lethal_probability": facts.lethal_probability * 5000,
                "future_bonus": future_bonus,
                "survival": survival * 0.25,
                "switch_value": switch_score,
                "objective": objective_score,
                "burst_setup": burst_setup,
                "punish_banking": punish,
                "desperation": desperation,
                "deny_burst": deny_burst,
                "expected_incoming": incoming,
                "passive_streak": self.passive_streak,
            }
            outcome = self.tactical_outcome(state, move, belief, facts)
            components["guaranteed_lethal"] = outcome.guaranteed_lethal
            components["kill_and_defend"] = outcome.kill_and_defend
            components["guaranteed_immediate_loss"] = outcome.guaranteed_immediate_loss
            components["loss_probability"] = outcome.loss_probability
            scored.append((score, move, facts, components, outcome))
        match_winners = [item for item in scored if item[4].wins_match]
        if match_winners:
            scored = match_winners
        else:
            fully_safe = [item for item in scored
                          if item[4].loss_probability <= 0.0]
            if fully_safe:
                scored = fully_safe
            else:
                safe = [item for item in scored
                        if not item[4].guaranteed_immediate_loss]
                if safe:
                    scored = safe
        scored.sort(key=lambda item: (
            item[4].wins_match,
            not item[4].guaranteed_immediate_loss,
            -item[4].loss_probability,
            item[4].kill_and_defend,
            item[4].guaranteed_lethal,
            item[4].lethal_probability,
            item[0],
        ), reverse=True)
        if self.temperature > 0 and self.rng is not None:
            # Mix only at genuine decision points: a tight value band around
            # the best move, and only when at least two moves are inside it.
            def _dominated(item):
                """A switch to a strictly weaker body wins nothing, even in
                theory: the same allocation without the switch is strictly
                better."""
                if not item[1].switch:
                    return False
                sv = switch_by_target.get(item[1].switch_to)
                return sv is not None and sv.value < 0

            candidates = [item for item in scored
                          if item[0] >= 0 and not _dominated(item)] or scored
            top = max(item[0] for item in candidates)
            band = self.band_fraction * (top if top > 0 else 1.0)
            tie_set = [item for item in candidates if item[0] >= top - band]
            if len(tie_set) >= 2:
                tie_scores = [item[0] for item in tie_set]
                temp = self.temperature * (top if top > 0 else 1.0)
                weights = [math.exp((s - top) / temp) for s in tie_scores]
                total = sum(weights)
                pick = self.rng.random() * total
                acc = 0.0
                idx = len(tie_set) - 1
                for i, w in enumerate(weights):
                    acc += w
                    if pick <= acc:
                        idx = i
                        break
                score, move, facts, components, outcome = tie_set[idx]
            else:
                score, move, facts, components, outcome = scored[0]
        else:
            score, move, facts, components, outcome = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else score
        self.last_report = {
            "objective": self.objective.name,
            "objective_reason": self.objective.reason,
            "belief": belief.probabilities,
            "selected": move.label,
            "facts": facts.__dict__,
            "tactical_outcome": outcome.__dict__,
            "score_components": components,
            "move_quality": {
                "rank": 1,
                "score": score,
                "best_score": score,
                "regret_vs_best": 0.0,
                "margin_over_second": score - second_score,
            },
            "alternatives": [(item[1].label, item[0], item[3]) for item in scored[:8]],
            "switch_values": [value.__dict__ for value in switch_values],
            "opponent_allocations": [
                {"move": item.allocation.label, "probability": item.probability}
                for item in self.opponent_allocations(state)[:12]
            ],
            "policy_belief": self.history.policy_belief,
        }
        self.passive_streak = self.passive_streak + 1 if move.attacks == 0 else 0
        return move

    def _expected_incoming(self, state: GameState, shields: int) -> float:
        """Estimate incoming damage from a public-history allocation distribution."""
        enemy = state.opponent.active_character
        return sum(
            exchange_damage(
                enemy, state.player.active_character,
                max(0, hypothesis.allocation.attacks - shields))
            * hypothesis.probability
            for hypothesis in self.opponent_allocations(state)
        )

    def _objective_score(self, move, facts, survival, active_hp):
        if self.objective.name == "finish":
            return facts.lethal_probability * 3500 + (2000 if facts.guaranteed_lethal else 0)
        if self.objective.name == "survive":
            return survival * 1.2 + move.defends * 500 - facts.expected_damage * 0.2
        if self.objective.name == "prepare_burst":
            return move.bonuses * 900 + move.defends * 350 - move.attacks * 80
        if self.objective.name == "deny_burst":
            return move.defends * 1000 + move.bonuses * 450 - move.attacks * 120
        if self.objective.name == "break_stall":
            return move.attacks * 1200 + move.bonuses * 700 - move.defends * 150
        return 0.0

    def _search(self, state, depth, root_turn):
        key = (state, depth, root_turn)
        if key in self._search_cache:
            return self._search_cache[key]
        if state.player.lost or state.opponent.lost:
            value = self.evaluate(state, depth)
            self._search_cache[key] = value
            return value
        if depth <= 0:
            terminal_values = [
                self.evaluate(child, depth)
                for move in legal_allocations(
                    state.player if state.player_to_move else state.opponent)
                if (child := apply(state, move)).player.lost or child.opponent.lost
            ]
            if terminal_values:
                value = (max(terminal_values) if state.player_to_move
                         else min(terminal_values))
            else:
                value = self.evaluate(state, depth)
            self._search_cache[key] = value
            return value
        values = []
        side = state.player if state.player_to_move else state.opponent
        for move in legal_allocations(side):
            values.append(self._search(apply(state, move), depth - 1, root_turn))
        value = ((max(values) if state.player_to_move == root_turn else min(values))
                 if values else self.evaluate(state, depth))
        self._search_cache[key] = value
        return value

    def evaluate(self, state, depth=0):
        """Static positional value, with the opponent's bank supplied by belief."""
        if state.opponent.lost:
            return 100000 + max(0, depth) * 100
        if state.player.lost:
            return -100000 - max(0, depth) * 100
        material = (sum(c.hp for c in state.player.characters)
                    - sum(c.hp for c in state.opponent.characters)) * 1.5
        bodies = (state.player.alive_count - state.opponent.alive_count) * 2600
        believed_bank = max(state.opponent.bonus, self._believed_bank)
        bonus = (state.player.bonus - believed_bank) * 450

        def pressure(actor: Side, target: Side, budget: int) -> float:
            best = 0
            for index, character in enumerate(actor.characters):
                if not character.alive:
                    continue
                attacks = max(0, budget - (index != actor.active) - target.shields)
                best = max(best, exchange_damage(
                    character, target.active_character, attacks))
            lethal = 3500 if best >= target.active_character.hp else 0
            return min(best, target.active_character.hp) + lethal

        if state.player_to_move:
            player_budget = state.player.actions
            opponent_budget = min(
                MAX_ACTIONS, base_budget(state.turn + 1) + believed_bank)
        else:
            player_budget = next_budget(state.turn + 1, state.player)
            opponent_budget = state.opponent.actions
        tempo = pressure(state.player, state.opponent, player_budget)
        tempo -= pressure(state.opponent, state.player, opponent_budget)
        return material + bodies + bonus + tempo * 0.7


# =========================================================================
# interactive.py (pieces used by the harness)
# =========================================================================
EMOJI = {Type.A: "[A]", Type.B: "[B]", Type.C: "[C]", Type.D: "[D]"}
TYPE_NAMES = {Type.A: "Artist", Type.B: "Brawler", Type.C: "Coordinator", Type.D: "Defender"}


class PlayerQuit(Exception):
    pass


def bar(value, maximum, width=12):
    filled = int(max(0, min(value / max(1, maximum), 1.0)) * width)
    return "#" * filled + "." * (width - filled)


def _available_moves(state):
    return legal_allocations(state.player)


def _parse_switch(raw):
    try:
        return int(raw) - 1
    except ValueError:
        return None


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


# =========================================================================
# play.py harness
# =========================================================================
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
        if tag == "YOU":
            lines.append(f"   your shields: {side.shields}")
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
           "bot": _side_dict(bot),
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
        print("SESSION COMPLETE — all games recorded. Run:")
        print("  python track_winrate.py show")
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
    print("\nSESSION COMPLETE — all games recorded. Run:")
    print("  python track_winrate.py show")


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
    else:
        p.print_help()


if __name__ == "__main__":
    main()