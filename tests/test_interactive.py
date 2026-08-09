import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from cote_megaverse.interactive import (
    bar, human_allocation, parse_allocation, parse_team, show_resolution,
    show_outcome, show_state,
)
from cote_megaverse.rules import Type, initial


class InteractiveTests(unittest.TestCase):
    def test_historical_bar(self):
        self.assertEqual(bar(6300, 6300), "############")
        self.assertEqual(bar(0, 6300), "............")

    def test_parse_team_and_allocation(self):
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        self.assertEqual(parse_team("A, B, C"), (Type.A, Type.B, Type.C))
        self.assertEqual(parse_allocation("1,0,0", state).attacks, 1)

    def test_human_action_prompt_matches_historical_flow(self):
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        answers = iter(["a"])
        move = human_allocation(state, input_fn=lambda _: next(answers), output_fn=lambda _: None)
        self.assertEqual(move.label, "a1/d0/b0")

    def test_state_hides_opponent_shields(self):
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        output = io.StringIO()
        with redirect_stdout(output):
            show_state(state)
        text = output.getvalue()
        self.assertNotIn("Shields", text)
        self.assertIn("Actions: 1 + 0", text)
        self.assertNotIn("KILL possible", text)

    def test_active_character_is_rendered_first(self):
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        state = state.__class__(
            state.player.__class__(state.player.characters, 2, (2, 0, 1), 0, 0, 1),
            state.opponent, state.turn, True,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            show_state(state)
        lines = [line for line in output.getvalue().splitlines() if "[A]#" in line or "[C]#" in line]
        self.assertTrue(lines[0].lstrip().startswith("> [C]#3"))

    def test_blocked_resolution_uses_block_label(self):
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        state = state.__class__(state.player, state.opponent.__class__(
            state.opponent.characters, state.opponent.active,
            state.opponent.stack_order, state.opponent.bonus, 2, 0),
            state.turn, True)
        output = io.StringIO()
        with redirect_stdout(output):
            show_resolution(state, parse_allocation("1,0,0", state), state, "AI")
        text = output.getvalue()
        self.assertIn("BLOCK!", text)
        self.assertNotIn("AI spent", text)
        self.assertIn("AI: 1 attacks vs 2 shields", text)
        self.assertNotIn("0 dmg", text)

    def test_ai_allocation_remains_hidden_after_its_turn(self):
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        output = io.StringIO()
        with redirect_stdout(output):
            show_resolution(state, parse_allocation("0,0,1", state), state, "AI")
        text = output.getvalue()
        self.assertNotIn("AI spent", text)
        self.assertNotIn("AI selected", text)

    def test_player_resolution_always_reports_ai_spent_shields(self):
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        output = io.StringIO()
        with redirect_stdout(output):
            show_resolution(state, parse_allocation("0,0,1", state), state, "You")
        text = output.getvalue()
        self.assertIn("AI spent 0 shields", text)
        self.assertIn("You did not attack", text)

    def test_kill_preview_ignores_opponent_shields(self):
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        player_characters = list(state.player.characters)
        player_characters[0] = replace(player_characters[0], hp=6000, max_hp=6000, atk=2000)
        opponent_characters = list(state.opponent.characters)
        opponent_characters[0] = replace(opponent_characters[0], hp=6000, max_hp=6000, atk=2000)
        state = state.__class__(
            state.player.__class__(tuple(player_characters), state.player.active,
                                   state.player.stack_order, state.player.bonus,
                                   state.player.shields, 7),
            state.opponent.__class__(tuple(opponent_characters), state.opponent.active,
                                     state.opponent.stack_order, state.opponent.bonus,
                                     2, state.opponent.actions),
            state.turn, True,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            show_state(state)
        self.assertIn("[KILL possible] 3 attacks needed", output.getvalue())

    def test_outcome_is_a_single_human_result(self):
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        defeated = tuple(replace(character, hp=0) for character in state.opponent.characters)
        state = replace(state, opponent=replace(state.opponent, characters=defeated))
        output = io.StringIO()
        winner = show_outcome(state, output.write)
        self.assertEqual(winner, "YOU")
        self.assertEqual(output.getvalue(), "\n  YOU WIN")
        self.assertNotIn("BATTLE OVER", output.getvalue())


if __name__ == "__main__":
    unittest.main()
