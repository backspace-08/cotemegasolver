"""Continual Subgame Resolving regression tests.

1. Reach conservation: the opponent reach profile stored after a resolve sums
   to 1.0 (or is None -> uniform fallback).
2. Turtling punishment: against a 1-HP full-shielder the bot must break the
   a4-into-d4 deadlock (bank -> a8) and win within a few rounds, instead of
   spamming a4 forever.
3. Lethal burst defense: when the opponent holds bank=4 and will burst a8, the
   bot must put >= 4 shields, not leave itself open.
"""
import os
import random
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from cote_megaverse.rules import (  # noqa: E402
    GameState, Side, Character, Type, apply)
from cote_megaverse.cfr_bot import CFRBot  # noqa: E402

HP, ATK = 24000, 1500


def make_side(t, hp=HP, shields=0):
    return Side(characters=(Character(type=t, hp=hp, atk=ATK, max_hp=hp),),
                active=0, stack_order=(0,), bonus=0, shields=shields, actions=0)


class ReachConservationTests(unittest.TestCase):
    def test_reach_conservation(self):
        bot = CFRBot(depth=3, iters=80, cap=6, gamma=0.995, prune_after=20,
                     temperature=1.0, rng=random.Random(1))
        state = GameState(player=make_side(Type.A), opponent=make_side(Type.A),
                          turn=7, player_to_move=True).prepare()
        planning = GameState(state.player, state.opponent, state.turn, True)
        bot.choose(planning)
        if bot._reach is not None:
            self.assertAlmostEqual(sum(bot._reach.values()), 1.0, places=6)


class TurtlingPunishmentTests(unittest.TestCase):
    def _run(self, seed):
        bot = CFRBot(depth=3, iters=120, cap=6, gamma=0.995, prune_after=20,
                     temperature=1.0, rng=random.Random(seed))
        # CFR 7500 (5 hits) vs PL 1500 (1 hit) holding d4 shields, CFR to move
        state = GameState(player=make_side(Type.A, 7500),
                          opponent=make_side(Type.A, 1500, shields=4),
                          turn=7, player_to_move=True).prepare()
        moves = []
        for _ in range(8):  # 4 full rounds
            if state.player.lost or state.opponent.lost:
                break
            if state.player_to_move:
                before = state
                planning = GameState(state.player, state.opponent, state.turn, True)
                move = bot.choose(planning)
                moves.append((state.turn, move.attacks, move.defends, move.bonuses))
                state = apply(state, move)
                bot.observe_shields(before.opponent.shields)
            else:
                before = state
                move = type("M", (), {"attacks": 0, "defends": 4, "bonuses": 0,
                                      "switch": False})()
                state = apply(state, move)
                bot.observe(move.attacks, move.bonuses, move.switch,
                            budget=before.opponent.actions)
        return state, moves

    def test_turtling_1hp_punishment(self):
        state, moves = self._run(seed=1)
        self.assertTrue(state.opponent.lost,
                        "CFR must kill the 1-HP turtler; moves=%s" % moves)
        # the win must NOT be pure a4-into-d4 spam: it needs bank-then-burst
        # (an attack that breaks through d4) or a banked build-up
        punished = [m for m in moves if m[1] >= 5 or m[3] >= 1]
        self.assertTrue(punished,
                        "CFR must bank->burst or attack through d4, "
                        "not a4-spam; moves=%s" % moves)


class LethalBurstDefenseTests(unittest.TestCase):
    def test_lethal_burst_defense(self):
        # PL at 1 HP, holds bank=4 (belief exact), will burst a8. CFR must shield.
        bot = CFRBot(depth=3, iters=120, cap=6, gamma=0.995, prune_after=20,
                     rng=None)  # argmax for determinism
        state = GameState(player=make_side(Type.A, 7500),
                          opponent=make_side(Type.A, 1500),
                          turn=7, player_to_move=True).prepare()
        # pin the belief to (sh=0, bank=4): opponent banked b4 publicly
        bot.observe(attacks=0, bonuses=4, switched=False, budget=8, turn=7)
        bot.model._restrict(lambda shields, bank: bank == 4)
        planning = GameState(state.player, state.opponent, state.turn, True)
        move = bot.choose(planning)
        self.assertGreaterEqual(move.defends, 4,
                                "must shield >=4 vs imminent a8 burst, got (%d,%d,%d)"
                                % (move.attacks, move.defends, move.bonuses))


if __name__ == "__main__":
    unittest.main()

