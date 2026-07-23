"""Human-fair belief-aware planning agent."""

from collections import defaultdict
from dataclasses import dataclass, field
from math import ceil

from .rules import (Allocation, GameState, MAX_ACTIONS, MAX_BONUS, Type, apply,
                     base_budget, legal_allocations, multiplier, next_budget,
                     rounded_damage)
from .strategy import Objective, switch_value


@dataclass
class PublicHistory:
    actions: list[tuple[int, int, int, int]] = field(default_factory=list)

    def observe(self, attacks: int, bonuses: int, switched: bool):
        self.actions.append((attacks, 0, bonuses, int(switched)))

    def observe_resolved(self, attacks, defends, bonuses, switched=False):
        self.actions.append((attacks, defends, bonuses, int(switched)))

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
class TacticalFacts:
    damage_per_hit: int
    hits_to_kill: int
    lethal_probability: float
    guaranteed_lethal: bool
    expected_damage: float


class Planner:
    def __init__(self, depth=3, branch_limit=None):
        self.depth = depth
        self.branch_limit = branch_limit
        self.history = PublicHistory()
        self.objective = Objective()
        self.last_report = {}

    def observe(self, attacks, bonuses, switched=False):
        self.history.observe(attacks, bonuses, switched)

    def belief(self, opponent: GameState | object) -> ShieldBelief:
        side = opponent.opponent if hasattr(opponent, "opponent") else opponent
        if hasattr(opponent, "turn"):
            budget_turn = opponent.turn + (1 if opponent.player_to_move else 0)
            budget = next_budget(budget_turn, side)
        else:
            budget = min(MAX_ACTIONS, side.actions)
        weights = {i: 1.0 for i in range(budget + 1)}
        if self.history.attack_rate > 0.6:
            weights = {i: value / (i + 1) for i, value in weights.items()}
        if self.history.defend_rate > 0.4:
            weights = {i: value * (1 + i * 0.5) for i, value in weights.items()}
        total = sum(weights.values())
        return ShieldBelief({i: value / total for i, value in weights.items()})

    def facts(self, state: GameState, allocation: Allocation, belief: ShieldBelief):
        me, enemy = state.player, state.opponent
        character = me.characters[allocation.switch_to if allocation.switch else me.active]
        target = enemy.active_character
        damage = rounded_damage(character.atk * multiplier(character.type, target.type))
        needed = ceil(target.hp / max(damage, 1))
        values = [max(0, allocation.attacks - shields) for shields in belief.probabilities]
        lethal = sum(p for shields, p in belief.probabilities.items() if allocation.attacks - shields >= needed)
        expected = sum(min(target.hp, max(0, allocation.attacks - shields) * damage) * p
                       for shields, p in belief.probabilities.items())
        return TacticalFacts(damage, needed, lethal, min(values) >= needed, expected)

    def choose(self, state: GameState) -> Allocation:
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
            turn=state.turn,
        )
        switch_by_target = {value.target: value for value in switch_values}
        scored = []
        for move, facts in candidate_facts:
            child = apply(state, move)
            continuation = self._search(child, self.depth - 1, state.player_to_move)
            incoming = self._expected_incoming(state, move.defends)
            survival = max(0.0, state.player.active_character.hp - incoming)
            future_bonus = min(MAX_BONUS, state.player.bonus + move.bonuses) * 300
            switch = switch_by_target.get(move.switch_to)
            switch_score = switch.value * 0.8 if switch else 0.0
            objective_score = self._objective_score(
                move, facts, survival, state.player.active_character.hp)
            score = continuation + facts.expected_damage * 0.6
            score += facts.lethal_probability * 5000 + future_bonus
            score += survival * 0.25 + switch_score + objective_score
            components = {
                "continuation": continuation,
                "expected_damage": facts.expected_damage * 0.6,
                "lethal_probability": facts.lethal_probability * 5000,
                "future_bonus": future_bonus,
                "survival": survival * 0.25,
                "switch_value": switch_score,
                "objective": objective_score,
                "expected_incoming": incoming,
            }
            scored.append((score, move, facts, components))
        scored.sort(key=lambda item: item[0], reverse=True)
        score, move, facts, components = scored[0]
        self.last_report = {
            "objective": self.objective.name,
            "objective_reason": self.objective.reason,
            "belief": belief.probabilities,
            "selected": move.label,
            "facts": facts.__dict__,
            "score_components": components,
            "alternatives": [(item[1].label, item[0], item[3]) for item in scored[:8]],
            "switch_values": [value.__dict__ for value in switch_values],
        }
        return move

    def _expected_incoming(self, state: GameState, shields: int) -> float:
        """Estimate next incoming damage without reading hidden opponent state."""
        enemy = state.opponent.active_character
        budget = next_budget(state.turn + 1, state.opponent)
        attacks = max(0.0, budget * self.history.attack_rate - shields)
        damage = rounded_damage(enemy.atk * multiplier(
            enemy.type, state.player.active_character.type))
        return attacks * damage

    def _objective_score(self, move, facts, survival, active_hp):
        if self.objective.name == "finish":
            return facts.lethal_probability * 3500 + (2000 if facts.guaranteed_lethal else 0)
        if self.objective.name == "survive":
            return survival * 1.2 + move.defends * 500 - facts.expected_damage * 0.2
        if self.objective.name == "prepare_burst":
            return move.bonuses * 900 + move.defends * 350 - move.attacks * 80
        return 0.0

    def _search(self, state, depth, root_turn):
        if depth <= 0 or state.player.lost or state.opponent.lost:
            return self.evaluate(state)
        values = []
        side = state.player if state.player_to_move else state.opponent
        for move in legal_allocations(side):
            values.append(self._search(apply(state, move), depth - 1, root_turn))
        return (max(values) if state.player_to_move == root_turn else min(values)) if values else self.evaluate(state)

    @staticmethod
    def evaluate(state):
        if state.opponent.lost:
            return 100000
        if state.player.lost:
            return -100000
        return (sum(c.hp for c in state.player.characters) - sum(c.hp for c in state.opponent.characters)) * 1.5 \
            + (state.player.alive_count - state.opponent.alive_count) * 1800 \
            + (state.player.bonus - state.opponent.bonus) * 350
