import os
import sys
import unittest
from dataclasses import replace
from random import Random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from cote_megaverse.agent import Planner
from cote_megaverse.rules import (ATK_POOL, HP_POOL, Allocation, Character,
                                   GameState, Side, Type, apply,
                                   attacks_to_kill, exchange_damage, initial,
                                   legal_allocations)


class NewEngineTests(unittest.TestCase):
    def test_team_stats_come_from_declared_pools(self):
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        for character in state.player.characters + state.opponent.characters:
            self.assertIn(character.hp, HP_POOL)
            self.assertIn(character.atk, ATK_POOL)
            self.assertEqual(character.max_hp, character.hp)

    def test_exchange_rounds_per_hit(self):
        state = initial((Type.D, Type.A, Type.C), (Type.C, Type.A, Type.D))
        attacker = replace(state.player.characters[0], atk=2100)
        defender = replace(state.opponent.characters[0], hp=6000, max_hp=6000)
        self.assertEqual(exchange_damage(attacker, defender, 1), 1500)
        self.assertEqual(exchange_damage(attacker, defender, 4), 6000)
        self.assertEqual(attacks_to_kill(attacker, defender, shields=0), 4)
        self.assertEqual(attacks_to_kill(attacker, defender, shields=2), 6)

    def test_seed_reproduces_team_stats(self):
        first = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D), rng=Random(17))
        second = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D), rng=Random(17))
        first_stats = [(c.hp, c.atk) for c in first.player.characters + first.opponent.characters]
        second_stats = [(c.hp, c.atk) for c in second.player.characters + second.opponent.characters]
        self.assertEqual(first_stats, second_stats)

    def test_budget_and_switch_branching(self):
        state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
        self.assertEqual(state.player.actions, 1)
        self.assertEqual(len(legal_allocations(state.player)), 5)

    def test_forced_promotion_uses_target_order_and_clean_stack(self):
        # When the active character dies, the next living character comes from
        # the TARGET's own stack order (never the actor's), and the rebuilt
        # stack has no duplicates and no dead entries.
        def ch(t, hp, atk):
            return Character(t, hp, atk, hp)

        actor = Side((ch(Type.B, 800, 2100), ch(Type.A, 6000, 1900),
                      ch(Type.C, 400, 2000)),
                     active=2, stack_order=(2, 0, 1), actions=4)
        target = Side((ch(Type.A, 1, 2000), ch(Type.D, 6000, 2100),
                       ch(Type.D, 6300, 1900)),
                      active=0, stack_order=(0, 1, 2))
        state = GameState(actor, target, turn=7, player_to_move=True)
        after = apply(state, Allocation(4, 0, 0, None))
        t = after.opponent
        self.assertEqual(t.active, 1)                 # target's own order -> D#1
        self.assertEqual(t.stack_order, (1, 2))        # clean, promoted first
        self.assertEqual(len(t.stack_order), len(set(t.stack_order)))  # no dupes
        self.assertTrue(all(t.characters[i].alive for i in t.stack_order))
        self.assertTrue(t.forced_promotion)

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
        self.assertIn("move_quality", planner.last_report)
        self.assertIn("attacks_for_guaranteed_lethal", planner.last_report["facts"])

    def test_legal_allocations_always_spend_budget(self):
        for turn in range(1, 9):
            state = initial((Type.A, Type.B, Type.C), (Type.B, Type.C, Type.D))
            state = state.__class__(state.player.__class__(
                state.player.characters, state.player.active, state.player.stack_order,
                state.player.bonus, state.player.shields, turn),
                state.opponent, turn, True).prepare()
            for move in legal_allocations(state.player):
                self.assertEqual(move.attacks + move.defends + move.bonuses + (1 if move.switch else 0), state.player.actions)

    def test_planner_avoids_immediate_endgame_loss(self):
        ai = (
            Character(Type.A, 0, 1900, 5800),
            Character(Type.D, 0, 2100, 5900),
            Character(Type.A, 6000, 2000, 6000),
        )
        human = (
            Character(Type.C, 0, 2000, 6300),
            Character(Type.A, 5800, 2100, 5800),
            Character(Type.D, 3800, 1900, 5900),
        )
        state = GameState(
            Side(ai, active=2, stack_order=(2, 1, 0), actions=4),
            Side(human, active=2, stack_order=(2, 1, 0)),
            turn=8,
            player_to_move=True,
        )
        move = Planner(depth=3).choose(state)
        self.assertGreaterEqual(move.defends, 2)
        self.assertLess(move.attacks, 4)

    def test_planner_keeps_shield_worlds_live_from_a_remainder(self):
        ai = (
            Character(Type.B, 3400, 2100, 5900),
            Character(Type.D, 0, 1900, 6000),
            Character(Type.B, 6300, 1900, 6300),
        )
        human = (
            Character(Type.A, 0, 1900, 6100),
            Character(Type.B, 3900, 2000, 6000),
            Character(Type.B, 5700, 2100, 5700),
        )
        state = GameState(
            Side(ai, active=0, stack_order=(0, 2, 1), actions=3),
            Side(human, active=1, stack_order=(1, 2, 0), shields=2),
            turn=6,
            player_to_move=True,
        )
        planner = Planner(depth=3)
        planner.observe(attacks=3, bonuses=0, switched=False, budget=5)
        move = planner.choose(state)
        # Budget 5 minus 3 public attacks leaves a remainder of 2, which split
        # into shields and bank in unknown proportion. All three worlds must
        # stay live; the resolver's true value of 2 must not leak in.
        belief = planner.last_report["belief"]
        self.assertEqual(set(belief), {0, 1, 2})
        self.assertAlmostEqual(sum(belief.values()), 1.0)
        self.assertIsNotNone(move)

    def test_planner_does_not_throw_attacks_into_cap_forced_shields(self):
        # The bank is capped at 4, so a remainder of 8 cannot be all bank: at
        # least 4 shields are held in every live world. Attacks 1..4 are then
        # certainly blocked and are pure waste. This is the one shield fact
        # that is provable from public information without an attack exchange.
        ai = (
            Character(Type.B, 0, 2100, 5900),
            Character(Type.D, 0, 1900, 6000),
            Character(Type.B, 6300, 1900, 6300),
        )
        human = (
            Character(Type.A, 0, 1900, 6100),
            Character(Type.B, 3900, 2000, 6000),
            Character(Type.B, 5700, 2100, 5700),
        )
        state = GameState(
            Side(ai, active=2, stack_order=(2, 0, 1), actions=4),
            Side(human, active=1, stack_order=(1, 2, 0), shields=4),
            turn=8,
            player_to_move=True,
        )
        planner = Planner(depth=3)
        planner.observe(attacks=0, bonuses=0, switched=False, budget=8)
        move = planner.choose(state)
        belief = planner.last_report["belief"]
        self.assertEqual(min(belief), 4)
        self.assertEqual(move.attacks, 0)

    def test_guaranteed_lethal_beats_preparation(self):
        player = (Character(Type.A, 6000, 2000, 6000),
                  Character(Type.B, 6000, 2000, 6000),
                  Character(Type.C, 6000, 2000, 6000))
        opponent = (Character(Type.A, 4000, 2000, 6000),
                    Character(Type.B, 6000, 2000, 6000),
                    Character(Type.C, 6000, 2000, 6000))
        state = GameState(
            Side(player, actions=2), Side(opponent), turn=2,
            player_to_move=True)
        planner = Planner(depth=1)
        planner.observe(attacks=0, bonuses=0, switched=False, budget=0)
        move = planner.choose(state)
        self.assertEqual(move.attacks, 2)
        self.assertTrue(planner.last_report["tactical_outcome"]["guaranteed_lethal"])

    def test_kill_plus_defense_beats_bare_kill(self):
        player = (Character(Type.A, 6000, 2000, 6000),
                  Character(Type.B, 6000, 2000, 6000),
                  Character(Type.C, 6000, 2000, 6000))
        opponent = (Character(Type.A, 4000, 2000, 6000),
                    Character(Type.B, 6000, 2000, 6000),
                    Character(Type.C, 6000, 2000, 6000))
        state = GameState(
            Side(player, actions=3), Side(opponent), turn=2,
            player_to_move=True)
        planner = Planner(depth=1)
        planner.observe(attacks=0, bonuses=0, switched=False, budget=0)
        move = planner.choose(state)
        self.assertEqual((move.attacks, move.defends), (2, 1))
        self.assertTrue(planner.last_report["tactical_outcome"]["kill_and_defend"])

    def test_opening_pressure_attacks_known_unshielded_target(self):
        player = (Character(Type.A, 6000, 2000, 6000),
                  Character(Type.B, 6000, 2000, 6000),
                  Character(Type.C, 6000, 2000, 6000))
        opponent = (Character(Type.B, 6000, 2000, 6000),
                    Character(Type.C, 6000, 2000, 6000),
                    Character(Type.D, 6000, 2000, 6000))
        state = GameState(Side(player, actions=1), Side(opponent), turn=1,
                          player_to_move=True)
        move = Planner(depth=1).choose(state)
        self.assertEqual(move.attacks, 1)


if __name__ == "__main__":
    unittest.main()
