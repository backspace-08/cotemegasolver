import os
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from cote_megaverse.agent import Planner, ShieldBelief
from cote_megaverse.benchmark import run_self_play
from cote_megaverse.observation import observe
from cote_megaverse.rules import Type, initial
from cote_megaverse.strategy import Objective, switch_value


class LayerTests(unittest.TestCase):
    def test_observation_hides_opponent_shields(self):
        state = initial((Type.A, Type.B, Type.C), (Type.A, Type.C, Type.D))
        state = state.__class__(state.player, state.opponent.__class__(
            state.opponent.characters, 0, 0, 2, 0), state.turn, state.player_to_move)
        public = observe(state)
        self.assertIsNone(public.opponent.shield_count)
        self.assertEqual(public.player.shield_count, 0)

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

    def test_attack_heavy_history_creates_prepare_burst_objective(self):
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        planner = Planner(depth=1)
        for _ in range(5):
            planner.history.observe_resolved(4, 0, 0)
        planner.choose(state)
        self.assertEqual(planner.objective.name, "prepare_burst")
        self.assertIn("objective", planner.last_report)

    def test_score_report_explains_survival_and_continuation(self):
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        planner = Planner(depth=1)
        planner.choose(state)
        self.assertIn("score_components", planner.last_report)
        self.assertIn("continuation", planner.last_report["score_components"])
        self.assertIn("expected_incoming", planner.last_report["score_components"])


if __name__ == "__main__":
    unittest.main()
