"""Opponent information set.

This module is the single authority on what one side may know about the other
side's two hidden variables:

- ``shields``: shields the opponent currently holds, which will block our next
  attacks and then expire;
- ``bank``: stored bonus the opponent currently holds, which will inflate their
  next action budget.

The two are not independent. Both are paid for out of the same public
remainder, so ``shields + bank == remainder``. A world where the opponent holds
many shields is a world where they banked little, and vice versa. The joint
distribution is therefore the object of interest; separate marginals would
throw away the correlation that makes the game readable.

Nothing here reads resolver secrets. Every input is a public fact: a revealed
budget, a resolved attack count, a visible switch, or a blocked-attack count.
"""

from dataclasses import dataclass, field

from .rules import MAX_ACTIONS, MAX_BONUS, base_budget


def legal_splits(remainder: int, capacity: int = MAX_BONUS):
    """Every legal ``(shields, bank)`` split of a public remainder.

    Banking is capped, so a large remainder forces shields. A remainder of 8
    with a cap of 4 proves at least 4 shields.
    """
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

    The model is a small weighted set of ``(shields, bank)`` candidates. Belief
    generation uses a max-entropy uniform prior (identical to the table
    builder's ``build_belief_roots``), narrowed only by public evidence: a
    revealed budget, a resolved attack count, or a blocked-attack count.
    """

    records: list[TurnRecord] = field(default_factory=list)
    _candidates: dict[tuple[int, int], float] = field(
        default_factory=lambda: {(0, 0): 1.0})
    # Observation Consistency Filtering (persistent, survives observe_turn):
    # a fully/partially revealed attack pins or lower-bounds the shields of
    # the split that was attacked; a revealed budget pins the bank. Each is
    # tagged with the remainder it constrains and applied in `_prior` only to
    # a split of that same remainder, so evidence is never wiped by the prior
    # reset and a 0-damage attack forces belief to (R, 0).
    _shield_pin: tuple[int, int] | None = None   # (remainder, shields) exact
    _shield_lb: tuple[int, int] | None = None    # (remainder, min_shields)
    _bank_pin: tuple[int, int] | None = None     # (remainder, bank)

    # ---------------------------------------------------------------- observe

    def observe_turn(self, turn: int, budget: int, attacks: int,
                     switched: bool = False):
        """Record a completed opponent allocation from public facts only.

        ``budget`` is public at the opponent's turn start. Because turn
        preparation always drains the whole bank, the budget retroactively
        proves the bank they were holding, which in turn fixes the split of the
        earlier remainder. That is the delayed reveal: an action banked on turn
        ``N`` becomes public on turn ``N + 2``.
        """
        revealed_bank = max(0, budget - base_budget(turn))
        self._confirm_previous_bank(revealed_bank)
        remainder = max(0, budget - attacks - int(switched))
        self.records.append(TurnRecord(
            turn=turn, budget=budget, attacks=attacks, switched=switched,
            remainder=remainder, revealed_bank=revealed_bank))
        # A fresh remainder replaces the previous hidden state entirely: old
        # shields have expired and the old bank was just spent.
        self._candidates = self._prior(remainder)

    def observe_our_attack(self, attacks: int, blocked: int):
        """Record what our own attack learned about their shields.

        ``blocked < attacks`` pins the shield count exactly. ``blocked ==
        attacks`` only proves a lower bound. Either way the surviving worlds
        also fix the bank, because the split must sum to the remainder.

        Callers must invoke this while the candidates still describe the
        opponent's LAST remainder (i.e. before observing the opponent's next
        turn): the shields our attack met were placed on that last turn, so they
        pin its split. Driving harnesses pass the defender's current shields
        (``before.<defender>.shields``) right after the attacker's move.
        """
        if attacks <= 0:
            return
        if blocked < attacks:
            self._restrict(lambda shields, bank: shields == blocked)
            if self.records:
                self._shield_pin = (self.records[-1].remainder, blocked)
                self._shield_lb = None
        else:
            self._restrict(lambda shields, bank: shields >= attacks)
            if self.records:
                self._shield_lb = (self.records[-1].remainder, attacks)
                self._shield_pin = None

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

    # ---------------------------------------------------------------- helpers

    def _prior(self, remainder: int) -> dict[tuple[int, int], float]:
        """Max-entropy uniform prior over the splits of ``remainder``, hard-
        filtered by observation-consistency facts.

        Every legal split carries weight 1/(R+1) - strictly identical to the
        table builder's ``build_belief_roots``. THEN any split inconsistent
        with a revealed shield pin / lower bound or a revealed bank is hard-
        zeroed (likelihood = 0) and the rest renormalized - e.g. an a4 that
        deals 0 damage against R=4 forces belief to (4, 0) exactly.
        """
        splits = legal_splits(remainder)
        if len(splits) == 1:
            return {splits[0]: 1.0}
        w = 1.0 / len(splits)
        cand = {s: w for s in splits}
        if self._shield_pin is not None and self._shield_pin[0] == remainder:
            cand = {k: v for k, v in cand.items() if k[0] == self._shield_pin[1]}
        elif self._shield_lb is not None and self._shield_lb[0] == remainder:
            cand = {k: v for k, v in cand.items() if k[0] >= self._shield_lb[1]}
        if self._bank_pin is not None and self._bank_pin[0] == remainder:
            cand = {k: v for k, v in cand.items() if k[1] == self._bank_pin[1]}
        if not cand:
            return {s: w for s in splits}
        tot = sum(cand.values())
        return {k: v / tot for k, v in cand.items()}

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
        prev_rem = self.records[-1].remainder
        self._restrict(lambda shields, bank: bank == revealed_bank)
        consistent = [key for key in self._candidates if key[1] == revealed_bank]
        if len(consistent) == 1:
            shields = consistent[0][0]
            self.records[-1] = TurnRecord(
                **{**self.records[-1].__dict__, "confirmed_shields": shields})
        if consistent:
            self._bank_pin = (prev_rem, revealed_bank)
