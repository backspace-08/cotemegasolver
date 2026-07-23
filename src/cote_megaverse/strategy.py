"""Strategic objectives and switch value calculations."""

from dataclasses import dataclass

from .rules import GameState, Type, multiplier, rounded_damage


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


def switch_value(state: GameState, target_index: int, belief) -> SwitchValue:
    me, enemy = state.player, state.opponent
    current = me.active_character
    target = me.characters[target_index]
    enemy_character = enemy.active_character
    current_damage = rounded_damage(current.atk * multiplier(current.type, enemy_character.type))
    target_damage = rounded_damage(target.atk * multiplier(target.type, enemy_character.type))
    current_received = rounded_damage(enemy_character.atk * multiplier(enemy_character.type, current.type))
    target_received = rounded_damage(enemy_character.atk * multiplier(enemy_character.type, target.type))
    value = target_damage - current_damage + current_received - target_received - 200
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
               expected_incoming: int, attack_rate: float, turn: int):
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
