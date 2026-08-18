"""Human-fair belief-aware planning agent."""

import math

from dataclasses import dataclass, field, replace

from .infoset import OpponentModel
from .rules import (Allocation, GameState, MAX_ACTIONS, MAX_BONUS, Side, apply,
                     attacks_to_kill, base_budget, exchange_damage,
                     legal_allocations, next_budget)
from .strategy import Objective, marginal_bonus_value, switch_value


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
        """Record one opponent allocation from PUBLIC facts only.

        ``bonuses`` is the opponent's hidden split and is deliberately never
        used: deriving a shield count from it (budget - attacks - bonuses)
        would leak the exact hidden shields. What is public is the budget, the
        attack count and a visible switch; the remainder (defends + bank) is
        ambiguous until the resolution reveals the defender's shields.
        """
        if budget is None:
            budget = attacks + int(switched)
        remainder = max(0, budget - attacks - int(switched))
        self.actions.append((attacks, remainder, 0, int(switched)))
        self.held_shields = None
        self.events.append(PublicEvent(attacks, 0, int(switched), None))
        self._update_policy(attacks, remainder, 0)

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
    # Probability mass of the believed worlds in which the opponent's next
    # allocation can end the match. `guaranteed_immediate_loss` is the special
    # case where this covers every live world. Tracking the partial case lets
    # the planner prefer a move that dies in no world over one that dies in a
    # credible few, as required by the safety-gate rule in AGENT.md §9.
    loss_probability: float = 0.0


class Planner:
    def __init__(self, depth=3, branch_limit=None, temperature=0.0, rng=None,
                 band_fraction=0.15, max_nodes=2000):
        self.depth = depth
        self.branch_limit = branch_limit
        # Context-conditional mixing. 0 -> fully deterministic argmax (fast,
        # reproducible, strong, good for tests). > 0 -> the bot mixes ONLY at
        # genuine decision points: when at least two moves score within
        # `band_fraction` of the best, it samples among them with a softmax of
        # width `temperature` (fraction of the best score). Everywhere else a
        # clear leader exists and is played deterministically, so mixing never
        # produces blunders and the bot keeps its positional strength.
        self.temperature = temperature
        self.band_fraction = band_fraction
        self.rng = rng
        self.history = PublicHistory()
        self.model = OpponentModel()
        self.objective = Objective()
        self.last_report = {}
        self._search_cache = {}
        self.passive_streak = 0
        # Credible opponent bank for evaluation, refreshed each `choose` from
        # OpponentModel. Kept as state (not a parameter) so the existing
        # `_search`/`evaluate` call sites stay unchanged. It is public
        # inference, never a resolver read.
        self._believed_bank = 0
        # Node budget for the depth-limited search. Late-game states with a
        # large budget blow up (all legal allocations x shield worlds at every
        # level); the budget cuts expansion deterministically (static eval),
        # bounding worst-case choose() time without touching decisions in
        # normal games.
        self.max_nodes = max_nodes
        self._nodes_used = 0

    def observe(self, attacks, bonuses=None, switched=False, budget=None, *,
                turn):
        """Record a public opponent allocation.

        ``bonuses`` is the opponent's hidden split and is deliberately ignored;
        it is accepted only so existing callers keep working. What is public is
        the budget, the attack count, and a visible switch. Everything else is
        a remainder that stays ambiguous until later evidence resolves it.
        ``turn`` is the half-turn the observed allocation happened on; required
        so ``base_budget(turn)`` computes the correct revealed bank.
        """
        if budget is None:
            budget = attacks + int(switched)
        self.model.observe_turn(turn, budget, attacks, switched)
        self.history.observe(attacks, bonuses or 0, switched, budget)

    def observe_shields(self, shields: int):
        """Our attack met this many shields, which pins the live worlds."""
        self.model.observe_our_attack(shields + 1, shields)
        self.history.reveal_latest_defends(shields)

    def observe_attack(self, attacks: int, blocked: int):
        """Report an attack we made with its exact blocked count.

        ``blocked < attacks`` pins the defender's shields exactly; ``blocked ==
        attacks`` (fully absorbed) only proves ``shields >= attacks``.

        Per RULES.md §8 the defender's shields are revealed in FULL on every
        resolution, so driving harnesses should call ``observe_shields`` (which
        pins the exact value). This method only records the lower-bound case and
        exists for legacy callers.
        """
        if attacks <= 0:
            return
        self.model.observe_our_attack(attacks, blocked)
        self.history.reveal_latest_defends(blocked)

    def belief(self, opponent: GameState | object) -> ShieldBelief:
        """Distribution over the opponent's currently held shields.

        Derived only from the public remainder and evidence that narrowed it.
        Never from ``state.opponent.shields`` or ``state.opponent.bonus``.
        """
        return ShieldBelief(self.model.shield_distribution())

    def opponent_budgets(self, state: GameState) -> dict[int, float]:
        """Distribution over the opponent's next public budget.

        Their bank is hidden, so their next budget is a distribution, not a
        number. Reading ``state.opponent.bonus`` here would be a leak.
        """
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
        values = [max(0, allocation.attacks - shields) for shields in belief.probabilities]
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
        """Value of banking `move.bonuses` now to kill on the bot's next turn.

        A concentrated burst is worth more than trickling damage into shields:
        it lands a lethal blow before the opponent can answer. Reward banking
        only when the resulting next-turn burst actually kills the active
        target against the believed shield worlds; otherwise it is just
        deferred damage and gets no bonus (avoid over-banking).
        """
        bank_after = move.bonuses
        if bank_after <= 0:
            return 0.0
        me, enemy = state.player, state.opponent
        attacker = me.characters[attacker_idx]
        target = enemy.active_character
        # Banking must not sacrifice survival. The opponent acts at turn+1 and
        # may kill the active character before the burst lands at turn+2. If the
        # worst-case incoming damage over the opponent's hidden bank kills the
        # active character, the burst never happens: return 0 (no reward).
        if attacker_idx == me.active:
            opp_turn = state.turn + 1
            worst_bank = max(self.model.bank_distribution().keys(), default=0)
            opp_budget = min(MAX_ACTIONS, base_budget(opp_turn) + worst_bank)
            incoming = exchange_damage(enemy.active_character, attacker,
                                       max(0, opp_budget - move.defends))
            if incoming >= attacker.hp:
                return 0.0
        my_next_turn = state.turn + 2
        next_budget = min(MAX_ACTIONS, base_budget(my_next_turn) + bank_after)
        for shields, _p in belief.probabilities.items():
            landed = max(0, next_budget - shields)
            if exchange_damage(attacker, target, landed) >= target.hp:
                return 5000.0 * bank_after
        return 0.0

    def _reply_kills_us(self, child: GameState, bank: int) -> bool:
        """Can the opponent's next allocation end the match against us?

        ``child`` is the position after our allocation resolved, so
        ``child.player`` is us and ``child.player.shields`` are the shields we
        just committed. ``bank`` is a *believed* bank from ``OpponentModel``,
        never a resolver read: it sets the reply budget we must survive.

        Only our active character can take damage in one allocation, so we can
        only be wiped when the active is our last living body. That makes the
        test a damage comparison instead of a full reply enumeration, which is
        both exact for this question and far cheaper.
        """
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
        """Evaluate hard tactical facts across every live joint world.

        Worlds are joint ``(shields, bank)`` hypotheses from the belief model,
        not shield marginals. The bank matters as much as the shields: it sets
        the budget of the reply we must survive. Masking the opponent's bank to
        zero (which ``choose`` must do for fairness) would otherwise make an
        eight-action burst invisible to this gate, which is exactly the
        bank-and-burst hole a human exploits.

        ``loses_in_some_world`` is tracked separately from
        ``guaranteed_immediate_loss``: §9 requires preferring a move that loses
        in no world, while only an all-worlds loss is unavoidable.
        """
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
        # shields or stored bank. Both are hidden; belief supplies them.
        state = replace(state, opponent=replace(state.opponent, shields=0,
                                                bonus=0))
        bank_belief = self.model.bank_distribution()
        # A burst threat is about the disaster branch, not the mean. Use the
        # largest bank that is still credible, so a possible spike is denied
        # rather than averaged away.
        credible_bank = max((bank for bank, p in bank_belief.items() if p >= 0.15),
                            default=0)
        # Feed the credible bank to `evaluate`, whose opponent-reply budget was
        # otherwise computed from the bank we just masked to zero, making every
        # banked burst invisible to the search.
        self._believed_bank = credible_bank
        self._search_cache.clear()
        self._nodes_used = 0
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

        # ---- Strategic layer signals -----------------------------------
        # (a) Punish banking: if the opponent is likely exposed (few shields)
        #     and is banking toward a burst, they are a sitting target RIGHT
        #     NOW. Pressure instead of mirroring their passivity.
        opponent_exposed = belief.probabilities.get(0, 0.0) >= 0.6
        opponent_banking = credible_bank >= 2
        # (b) All-in / threat when losing: if we are behind in bodies, a pure
        #     survival turtle just delays a loss. Prefer threat over shields.
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
            # Strategic layer adjustments.
            punish = 0.0
            if opponent_exposed and opponent_banking and move.attacks:
                punish = facts.expected_damage * 0.7 + move.attacks * 300
            desperation = 0.0
            if behind:
                if move.attacks:
                    desperation = facts.expected_damage * 0.4 + move.attacks * 150
                elif move.bonuses:
                    desperation = move.bonuses * 250.0      # set up a burst
                if move.attacks == 0 and move.defends > 0 and move.bonuses == 0:
                    desperation -= move.defends * 400.0      # pure turtle
            # deny_burst calibration: if the opponent is banking a big burst,
            # shields must be enough to survive its worst case — otherwise they
            # are wasted (the endgame turtle) and attacking/banking is better.
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
                        deny_burst = 450.0 * move.defends      # survives: good
                    else:
                        deny_burst = -450.0 * move.defends     # wasted: turtle
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
        # Gate precedence matters. A move that ends the match is unconditionally
        # best: nothing can follow it. A kill that merely removes one body is
        # not that, and must never override survival, because the opponent still
        # gets a reply. Safety is therefore checked before ordinary lethality.
        match_winners = [item for item in scored if item[4].wins_match]
        if match_winners:
            scored = match_winners
        else:
            # AGENT.md §9: "Prefer a move that loses in no world over a move
            # with a higher score that loses in one." Filter to fully safe moves
            # first; only if every move risks death do we fall back to the
            # least risky set, which is what `loss_probability` ranks below.
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
            # A clear leader is played deterministically; dominated switches
            # are never sampled (only reasoned).
            def _dominated(item):
                """A switch to a strictly weaker body wins nothing, even in
                theory: the same allocation without the switch is strictly
                better. Such switches must never be sampled, only reasoned."""
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
        self._nodes_used += 1
        if self._nodes_used > self.max_nodes:
            # Deterministic budget cutoff: stop expanding, use the static
            # value. Only reached in pathological late-game states.
            return self.evaluate(state, depth)
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
        """Static positional value, with the opponent's bank supplied by belief.

        ``choose`` masks ``state.opponent.bonus`` to zero for fairness, so this
        function must not read it to size the opponent's reply: doing so made
        every banked burst invisible to the search and was the mechanical cause
        of the bank-and-burst hole. ``self._believed_bank`` is the credible bank
        from ``OpponentModel`` (public inference), and deeper in the search the
        simulated bonus legitimately grows, so take whichever is larger.
        """
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
