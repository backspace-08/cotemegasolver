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


def _binomial(n: int, k: int) -> float:
    """Small binomial coefficient for split weighting."""
    if k < 0 or k > n:
        return 0.0
    result = 1.0
    for i in range(min(k, n - k)):
        result = result * (n - i) / (i + 1)
    return result


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

    The model is a small weighted set of ``(shields, bank)`` candidates. It is
    narrowed by evidence, never by assumption, and it is the only sanctioned
    source of opponent shield and bank values for the planner.
    """

    records: list[TurnRecord] = field(default_factory=list)
    _candidates: dict[tuple[int, int], float] = field(
        default_factory=lambda: {(0, 0): 1.0})
    _defend_count: float = 1.0
    _bank_count: float = 1.0
    _attack_actions: float = 0.0
    _total_actions: float = 0.0
    # Type hypothesis: the opponent is either a "full shielder", a "no
    # shielder" or "random split". Each fully-observed shield count (R, s) is
    # exact evidence (RULES.md §8); the posterior concentrates on the type that
    # explains the observed counts. This makes a persistent full-shielder reach
    # P(sh=R) ~0.96/0.99 after 2/3 fully-blocked rounds, where a plain
    # Beta-Binomial prior over per-action defend rate stays diffuse.
    _shield_samples: list[tuple[int, int]] = field(default_factory=list)

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
        self._attack_actions += attacks
        self._total_actions += max(1, budget)
        # A fresh remainder replaces the previous hidden state entirely: old
        # shields have expired and the old bank was just spent.
        first_round = len(self.records) == 1
        self._candidates = self._prior(remainder, first_round)

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
        else:
            self._restrict(lambda shields, bank: shields >= attacks)
        # Record the fully-revealed (remainder, shields) observation for the
        # type hypothesis. RULES.md §8: shields always fire/burn and are shown
        # exactly, so ``blocked < attacks`` pins the shield count precisely.
        if blocked < attacks and self.records:
            rem = self.records[-1].remainder
            if rem >= 0:
                self._shield_samples.append((rem, blocked))
        # The revealed split also feeds the behavioural prior (``defend_share``):
        # an opponent repeatedly observed holding no shields is a banker, so the
        # next remainder must lean toward bank rather than stay near-uniform.
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

    def _prior(self, remainder: int, first_round: bool = False) -> dict[tuple[int, int], float]:
        """Belief over the splits of ``remainder`` under the type hypothesis.

        Three opponent types compete: "full shield" (H1, every action goes to
        shields -> split (R, 0)), "no shield" (H0s, shields never used ->
        split (0, R)) and "random" (Hr, uniform over the R+1 splits). Each
        fully-revealed shield count (R, s) from RULES.md §8 is exact evidence;
        the posterior concentrates on the type that explains the observed
        counts. A persistent full-shielder pushes P(sh=R) to ~0.96/0.99 after
        2/3 rounds; a persistent banker concentrates on sh=0 the same way. A
        small uniform floor keeps every legal split live (AGENT.md §9: splits
        are pruned by public facts, never by assumption).

        ``first_round`` disables the full-shielder type: on the very first
        observed turn the opponent has not yet been attacked, so placing
        shields is strictly dominated (the equilibrium never plays d4 from a
        fresh board). The whole remainder must then be bank.
        """
        if remainder <= 0:
            return {(0, 0): 1.0}
        n_splits = remainder + 1
        p_full, p_none = self._type_posterior(remainder, first_round)
        p_rand = 1.0 - p_full - p_none
        EPSILON = 0.005
        weights: dict[tuple[int, int], float] = {}
        for shields, bank in legal_splits(remainder):
            if shields == remainder:
                weight = p_full + p_rand / n_splits
            elif shields == 0:
                weight = p_none + p_rand / n_splits
            else:
                weight = p_rand / n_splits
            weights[(shields, bank)] = weight + EPSILON
        return weights

    def _type_posterior(self, remainder: int, first_round: bool = False) -> tuple[float, float]:
        """(P(full-shielder), P(no-shielder)) from observed shield counts."""
        prior_full = 0.0 if first_round else 0.4
        prior_none = 1.0 if first_round else 0.4
        if not self._shield_samples:
            return prior_full, prior_none
        like_full = 1.0
        like_none = 1.0
        like_rand = 1.0
        for r, s in self._shield_samples:
            # Full-shielder produces s == r always; no-shielder s == 0 always;
            # random produces s uniform over 0..r.
            like_full *= 1.0 if s == r else 0.0
            like_none *= 1.0 if s == 0 else 0.0
            like_rand *= 1.0 / (r + 1)
        if like_full == 0.0 and like_none == 0.0:
            return 0.0, 0.0
        pf = prior_full * like_full
        pn = prior_none * like_none
        pr = (1.0 - prior_full - prior_none) * like_rand
        total = pf + pn + pr
        return pf / total, pn / total

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
