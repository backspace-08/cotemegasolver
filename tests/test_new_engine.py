import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from cote_megaverse.agent import Planner
from cote_megaverse.rules import Allocation, Type, initial, legal_allocations


class NewEngineTests(unittest.TestCase):
    def test_budget_and_switch_branching(self):
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        self.assertEqual(state.player.actions, 1)
        self.assertEqual(len(legal_allocations(state.player)), 5)

    def test_forced_promotion_is_not_voluntary_switch(self):
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        state = state.__class__(state.player, state.opponent.__class__(
            state.opponent.characters, 0, 0, 0, 2), state.turn, state.player_to_move)
        state = state.__class__(state.player, state.opponent, state.turn, False)
        state = state.__class__(state.player, state.opponent, 2, True)
        self.assertFalse(state.opponent.voluntary_switch_used)

    def test_planner_reports_belief_and_alternatives(self):
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        planner = Planner(depth=2)
        move = planner.choose(state)
        self.assertIsNotNone(move)
        self.assertIn("belief", planner.last_report)
        self.assertTrue(planner.last_report["alternatives"])


if __name__ == "__main__":
    unittest.main()
