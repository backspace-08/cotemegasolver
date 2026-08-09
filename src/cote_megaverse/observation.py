"""Public observation model. Hidden resolver fields never cross this boundary.

The stored bonus bank is hidden while it is stored. It becomes public only when
it is spent, because turn preparation drains the whole bank into the action
budget, so `bank = budget - base actions` is provable on the owner's next turn.
An action banked on turn `N` is therefore visible on turn `N + 2`.

This module reports structural bounds only. Narrowing the hidden bank using
public action accounting is the job of `infoset.OpponentModel`.
"""

from dataclasses import dataclass

from .rules import (MAX_ACTIONS, MAX_BONUS, Character, GameState, Side,
                    base_budget)


@dataclass(frozen=True)
class PublicSide:
    characters: tuple[Character, ...]
    active: int
    stack_order: tuple[int, ...]
    bonus: int | None
    actions: int
    shield_count: int | None = None

    @property
    def active_character(self):
        return self.characters[self.active]

    @property
    def alive_count(self):
        return sum(character.alive for character in self.characters)


@dataclass(frozen=True)
class PublicObservation:
    """Everything one side may know before the next allocation."""

    player: PublicSide
    opponent: PublicSide
    turn: int
    player_to_move: bool

    @property
    def opponent_is_acting(self):
        return not self.player_to_move

    @property
    def opponent_revealed_bank(self) -> int | None:
        """Bank the opponent just drained, provable only while it is acting."""
        if not self.opponent_is_acting:
            return None
        return max(0, self.opponent.actions - base_budget(self.turn))

    @property
    def opponent_bank_bounds(self) -> tuple[int, int]:
        """Structural bounds on the opponent's hidden stored bank."""
        revealed = self.opponent_revealed_bank
        if revealed is not None:
            # Preparation drained the whole bank, so nothing is stored now.
            return (0, 0)
        return (0, MAX_BONUS)

    @property
    def opponent_next_budget_bounds(self) -> tuple[int, int]:
        """Bounds on the opponent's budget on its next turn.

        Exact prediction is impossible: the bank is hidden, so this returns a
        range. Callers that need a distribution must use a belief model over
        remainders rather than a single number.
        """
        turn = self.turn + (1 if self.player_to_move else 2)
        base = base_budget(turn)
        low, high = self.opponent_bank_bounds
        if self.opponent_is_acting:
            # It may bank up to the cap out of the budget it is spending now.
            high = min(MAX_BONUS, self.opponent.actions)
        return (min(MAX_ACTIONS, base + low), min(MAX_ACTIONS, base + high))


def observe(state: GameState, for_player: bool = True) -> PublicObservation:
    """Build the observation for one side. Never leaks the other side's secrets."""
    me = state.player if for_player else state.opponent
    them = state.opponent if for_player else state.player

    def public(side: Side, hidden: bool):
        return PublicSide(
            characters=side.characters,
            active=side.active,
            stack_order=side.normalized_order(),
            bonus=None if hidden else side.bonus,
            actions=side.actions,
            shield_count=None if hidden else side.shields,
        )

    return PublicObservation(
        player=public(me, False),
        opponent=public(them, True),
        turn=state.turn,
        player_to_move=state.player_to_move if for_player else not state.player_to_move,
    )
