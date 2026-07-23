"""Public observation model. Hidden resolver fields never cross this boundary."""

from dataclasses import dataclass

from .rules import Character, GameState, Side, Type, next_budget


@dataclass(frozen=True)
class PublicSide:
    characters: tuple[Character, ...]
    active: int
    bonus: int
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
    """Everything a human can know before the next allocation."""

    player: PublicSide
    opponent: PublicSide
    turn: int
    player_to_move: bool

    @property
    def opponent_next_budget(self):
        side = Side(self.opponent.characters, self.opponent.active,
                    self.opponent.bonus, 0, self.opponent.actions)
        turn = self.turn + (1 if self.player_to_move else 0)
        return next_budget(turn, side)


def observe(state: GameState) -> PublicObservation:
    def public(side: Side, hidden: bool):
        return PublicSide(
            characters=side.characters,
            active=side.active,
            bonus=side.bonus,
            actions=side.actions,
            shield_count=None if hidden else side.shields,
        )

    return PublicObservation(
        player=public(state.player, False),
        opponent=public(state.opponent, True),
        turn=state.turn,
        player_to_move=state.player_to_move,
    )
