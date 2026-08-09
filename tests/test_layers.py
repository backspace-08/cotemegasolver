import os
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from cote_megaverse.agent import Planner, ShieldBelief
from cote_megaverse.benchmark import benchmark_policies, run_match, run_self_play
from cote_megaverse.infoset import OpponentModel
from cote_megaverse.observation import observe
from cote_megaverse.rules import Type, base_budget, initial
from cote_megaverse.strategy import Objective, marginal_bonus_value, switch_value


class LayerTests(unittest.TestCase):
    def test_observation_hides_opponent_shields(self):
        state = initial((Type.A, Type.B, Type.C), (Type.A, Type.C, Type.D))
        state = state.__class__(state.player, state.opponent.__class__(
            state.opponent.characters, 0, 0, 2, 0), state.turn, state.player_to_move)
        public = observe(state)
        self.assertIsNone(public.opponent.shield_count)
        self.assertEqual(public.player.shield_count, 0)

    def test_planner_choice_cannot_depend_on_exact_opponent_shields(self):
        state = initial((Type.A, Type.B, Type.C), (Type.A, Type.C, Type.D))
        no_shields = replace(state, opponent=replace(state.opponent, shields=0))
        many_shields = replace(state, opponent=replace(state.opponent, shields=4))
        first = Planner(depth=3).choose(no_shields)
        second = Planner(depth=3).choose(many_shields)
        self.assertEqual(first, second)

    def test_planner_only_learns_resolved_shields(self):
        planner = Planner(depth=1)
        planner.observe(attacks=1, bonuses=2, switched=False)
        self.assertEqual(planner.history.actions[-1], (1, 0, 2, 0))
        planner.observe_shields(3)
        self.assertEqual(planner.history.actions[-1], (1, 3, 2, 0))

    def test_public_budget_yields_a_remainder_not_an_exact_shield_count(self):
        # Budget 5 with 3 public attacks leaves a remainder of 2. That remainder
        # went to defends or bank in unknown proportion, so three worlds stay
        # live. Collapsing it to "2 shields" is the bug that made the bot
        # readable.
        planner = Planner(depth=1)
        planner.observe(attacks=3, bonuses=0, switched=False, budget=5)
        state = initial((Type.A, Type.B, Type.C), (Type.A, Type.C, Type.D))
        belief = planner.belief(state)
        self.assertEqual(set(belief.probabilities), {0, 1, 2})
        self.assertAlmostEqual(sum(belief.probabilities.values()), 1.0)

    def test_full_budget_attack_proves_a_zero_remainder(self):
        # Spending every action on attacks is self-revealing: nothing is left
        # for shields or bank, so belief becomes exact.
        planner = Planner(depth=1)
        planner.observe(attacks=3, bonuses=0, switched=False, budget=3)
        state = initial((Type.A, Type.B, Type.C), (Type.A, Type.C, Type.D))
        self.assertEqual(planner.belief(state).probabilities, {0: 1.0})

    def test_attack_evidence_pins_shields_exactly(self):
        # A partially blocked attack is the one fact that fixes the split.
        planner = Planner(depth=1)
        planner.observe(attacks=3, bonuses=0, switched=False, budget=5)
        planner.observe_shields(2)
        state = initial((Type.A, Type.B, Type.C), (Type.A, Type.C, Type.D))
        self.assertEqual(planner.belief(state).probabilities, {2: 1.0})

    def test_initial_shield_belief_is_exactly_zero(self):
        state = initial((Type.A, Type.B, Type.C), (Type.A, Type.C, Type.D))
        self.assertEqual(Planner(depth=1).belief(state).probabilities, {0: 1.0})

    def test_observation_hides_opponent_bank_and_gives_bounds_only(self):
        # The opponent's stored bank is hidden, so its next budget cannot be a
        # single number. Only bounds are public.
        state = initial((Type.A, Type.B, Type.C), (Type.A, Type.C, Type.D))
        state = replace(state, opponent=replace(state.opponent, bonus=3))
        public = observe(state)
        self.assertIsNone(public.opponent.bonus)
        low, high = public.opponent_next_budget_bounds
        expected_turn = state.turn + 1
        self.assertEqual(low, base_budget(expected_turn))
        self.assertGreater(high, low)

    def test_switch_value_prefers_advantageous_target(self):
        state = initial((Type.D, Type.A, Type.C), (Type.B, Type.C, Type.D))
        state = replace(state, player=replace(state.player, actions=2))
        value = switch_value(state, 1, ShieldBelief({0: 1.0}))
        self.assertTrue(value.recommended)
        self.assertGreater(value.target_damage, value.current_damage)

    def test_objective_finishes_on_guaranteed_lethal(self):
        objective = Objective()
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        objective.update(state, lethal_probability=1.0, expected_incoming=0,
                         attack_rate=0.1, turn=2)
        self.assertEqual(objective.name, "finish")

    def test_self_play_is_reproducible(self):
        first = run_self_play(seed=4, max_half_turns=8, depth=1)
        second = run_self_play(seed=4, max_half_turns=8, depth=1)
        self.assertEqual(first["winner"], second["winner"])
        self.assertEqual([item["move"] for item in first["replay"]],
                         [item["move"] for item in second["replay"]])
        self.assertEqual([item["player_to_move"] for item in first["replay"]],
                         [True, False, True, False, True, False, True, False])

    def test_attack_heavy_history_creates_prepare_burst_objective(self):
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        planner = Planner(depth=1)
        for _ in range(5):
            planner.history.observe_resolved(4, 0, 0)
        planner.choose(state)
        self.assertEqual(planner.objective.name, "prepare_burst")
        self.assertIn("objective", planner.last_report)

    def test_deny_burst_must_not_come_from_resolver_bank(self):
        # The stored bank is hidden. A planner handed a leaked resolver bank
        # must ignore it, otherwise it is reading a secret.
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        leaked = replace(state, opponent=replace(state.opponent, bonus=4))
        planner = Planner(depth=1)
        planner.choose(leaked)
        self.assertNotEqual(planner.objective.name, "deny_burst")

    def test_deny_burst_comes_from_a_credible_bank_belief(self):
        # Belief, not resolver state, is the sanctioned source of a burst read.
        objective = Objective()
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        objective.update(state, lethal_probability=0.0, expected_incoming=0,
                         attack_rate=0.1, turn=6, opponent_bank=3)
        self.assertEqual(objective.name, "deny_burst")

    def test_repeated_passivity_creates_break_stall_objective(self):
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        planner = Planner(depth=1)
        planner.passive_streak = 2
        planner.choose(state)
        self.assertEqual(planner.objective.name, "break_stall")

    def test_score_report_explains_survival_and_continuation(self):
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        planner = Planner(depth=1)
        planner.choose(state)
        self.assertIn("score_components", planner.last_report)
        self.assertIn("continuation", planner.last_report["score_components"])
        self.assertIn("expected_incoming", planner.last_report["score_components"])

    def test_policy_belief_and_marginal_bonus_are_reported(self):
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        planner = Planner(depth=1)
        planner.choose(state)
        self.assertAlmostEqual(sum(planner.last_report["policy_belief"].values()), 1.0)
        self.assertGreaterEqual(marginal_bonus_value(state, 1), 0)

    def test_human_policy_benchmark_is_seeded_and_measured(self):
        first = run_match(seed=3, policy="greedy", depth=1, max_half_turns=12)
        second = run_match(seed=3, policy="greedy", depth=1, max_half_turns=12)
        self.assertEqual(first["winner"], second["winner"])
        self.assertEqual([item["move"] for item in first["replay"]],
                         [item["move"] for item in second["replay"]])
        self.assertIn("missed_guaranteed_lethal", first["metrics"])
        self.assertIn("guaranteed_loss_moves", first["metrics"])
        self.assertEqual(first["metrics"]["ai_turns"],
                         first["metrics"]["human_turns"])

    def test_human_policy_benchmark_supports_ai_first(self):
        report = run_match(seed=3, policy="greedy", depth=1,
                           max_half_turns=8, ai_starts=True)
        self.assertTrue(report["ai_starts"])
        self.assertFalse(report["replay"][0]["player_to_move"])

    def test_policy_benchmark_returns_all_policies(self):
        report = benchmark_policies(seeds=range(2), depth=1, max_half_turns=4)
        self.assertEqual(set(report), {"random", "greedy", "bonus_shield"})
        self.assertEqual(report["greedy"]["games"], 4)


class BeliefSharpnessTests(unittest.TestCase):
    """The split prior must actually learn, while keeping every world live."""

    def test_prior_follows_observed_split_behaviour(self):
        # An opponent that spends its whole budget on attacks and bonuses, and
        # is then observed to have held zero shields, is a banker. The belief
        # must concentrate on "few shields", not stay near-uniform: a flat
        # prior is what stopped the planner from punishing a banking opponent.
        banker = OpponentModel()
        for turn in (1, 3, 5):
            banker.observe_turn(turn, base_budget(turn), attacks=0)
            banker.observe_our_attack(1, 0)
        banker.observe_turn(7, base_budget(7), attacks=0)
        distribution = banker.shield_distribution()
        self.assertGreater(distribution.get(0, 0.0), 0.5)
        self.assertGreater(distribution[0], distribution.get(4, 0.0) * 5)

    def test_prior_keeps_every_legal_split_live(self):
        # AGENT.md §9: splits are pruned by public facts, never by assumption.
        # However confident the behavioural read is, no legal world may be
        # assigned zero probability.
        banker = OpponentModel()
        for turn in (1, 3, 5):
            banker.observe_turn(turn, base_budget(turn), attacks=0)
            banker.observe_our_attack(1, 0)
        banker.observe_turn(7, base_budget(7), attacks=0)
        distribution = banker.shield_distribution()
        remainder = base_budget(7)
        for shields in range(remainder + 1):
            self.assertGreater(distribution.get(shields, 0.0), 0.0)


class LossGateTests(unittest.TestCase):
    """The safety gate must see bursts paid for out of a believed bank."""

    def test_reply_budget_uses_believed_bank_not_masked_state(self):
        # `choose` masks the opponent's bank to zero for fairness. If the loss
        # gate sized the opponent's reply from that masked value it would be
        # blind to every banked burst, which is the bank-and-burst hole.
        state = initial((Type.A,), (Type.A,))
        # 4 attacks (no bank) deal 7600, 8 attacks (bank 4) deal 15200, so an
        # HP of 9000 makes the no-bank reply survivable but the banked one
        # lethal — that is the flip the test must exercise.
        weak = replace(state.player.characters[0], hp=9000, atk=1900)
        state = replace(
            state,
            player=replace(state.player, characters=(weak,), stack_order=(0,)),
            opponent=replace(state.opponent, bonus=0),
            turn=7)
        planner = Planner(depth=1)
        # Same position, two different believed banks: a bigger bank buys a
        # bigger reply, so it must be able to flip the verdict.
        without_bank = planner._reply_kills_us(state, 0)
        with_bank = planner._reply_kills_us(state, 4)
        self.assertFalse(without_bank)
        self.assertTrue(with_bank)

    def test_loss_gate_ignores_positions_with_spare_bodies(self):
        # Only the active character can be damaged in one allocation, so a side
        # with a spare body cannot be wiped by a single reply.
        state = initial((Type.A, Type.A, Type.A), (Type.A, Type.A, Type.A))
        planner = Planner(depth=1)
        self.assertFalse(planner._reply_kills_us(state, 4))


if __name__ == "__main__":
    unittest.main()
