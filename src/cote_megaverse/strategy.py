"""Strategic objectives and switch value calculations."""

from dataclasses import dataclass

from .rules import GameState, exchange_damage, next_budget


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
        # The opponent's stored bank is hidden. Callers must pass a belief
        # estimate; ``state.opponent.bonus`` is a resolver secret and is only
        # used as a fallback for legacy callers that already masked it.
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
