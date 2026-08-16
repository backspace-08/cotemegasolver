"""CFR-based bot: plays through the Rust micro-tree re-solver.

The bot mirrors the Planner interface (choose/observe/observe_shields) so the
same harness can drive it head-to-head. It keeps an OpponentModel for the
belief over the opponent's hidden (shields, bank) split (public inference only,
never a resolver read), then re-solves the subgame from that belief with
``_cote_cfr.solve_micro_belief`` and plays the root strategy.

Leaf values: exact 1v1 equilibrium from the table for duel leaves, material
heuristic elsewhere (2v2/3v3). Depth/cap are configurable.
"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(BASE, "..", ".."))
_TABLE = os.path.join(_REPO, "cote_cfr", "1v1_table.csv")

import _cote_cfr  # noqa: E402

try:
    _cote_cfr.load_1v1_table(_TABLE)
except Exception as exc:  # pragma: no cover - best-effort table load
    print("cfr_bot: cannot load 1v1 table (%s); duel leaves fall back to material" % exc,
          file=sys.stderr)

from .infoset import OpponentModel  # noqa: E402
from .rules import Allocation, base_budget  # noqa: E402


def _team_of(side):
    return [(c.type.value, c.atk, c.max_hp // 10) for c in side.characters]


def _encode_state(state, opp_bank, opp_sh):
    """GameState -> flat Rust state, with the opponent split supplied by belief.

    [lenA, orderA..., hpA..., bankA, shA, lenB, orderB..., hpB..., bankB, shB,
     turn, to_move]; HP in units of 10, order active-first.

    The acting side's bank is drained into its prepared budget by ``prepare()``
    (bonus -> 0, actions = base + bonus), so recover it as actions - base(turn)
    to match the Rust trunk's budget = base + bank.
    """
    p, o = state.player, state.opponent
    order_a = list(p.normalized_order())
    hp_a = [p.characters[i].hp // 10 for i in order_a]
    order_b = list(o.normalized_order())
    hp_b = [o.characters[i].hp // 10 for i in order_b]
    bank_a = p.actions - base_budget(state.turn)
    return ([len(order_a)] + order_a + hp_a + [bank_a, p.shields]
            + [len(order_b)] + order_b + hp_b + [opp_bank, opp_sh]
            + [state.turn, 0 if state.player_to_move else 1])


class CFRBot:
    def __init__(self, depth=3, iters=400, cap=6, gamma=0.995,
                 temperature=0.0, rng=None, value_leaf=None):
        self.depth = depth
        self.iters = iters
        self.cap = cap
        self.gamma = gamma
        self.temperature = temperature
        self.rng = rng
        # optional belief-conditioned value network for the leaves (ReBeL/DeepStack)
        self.value_leaf = value_leaf
        self.model = OpponentModel()
        self._observed_turns = 0
        self.last_report = {}

    # ------------------------------------------------------------ observations
    def observe(self, attacks, bonuses=None, switched=False, budget=None,
                turn=None):
        """Record a public opponent allocation (mirrors Planner.observe)."""
        if turn is None:
            turn = self._observed_turns * 2 + 1
        self._observed_turns += 1
        if budget is None:
            budget = attacks + int(switched)
        self.model.observe_turn(turn, budget, attacks, switched)

    def observe_shields(self, shields):
        """Our attack met this many shields (mirrors Planner.observe_shields)."""
        self.model.observe_our_attack(shields + 1, shields)

    def observe_attack(self, attacks, blocked):
        """Report an attack we made with its exact blocked count.

        Per RULES.md §8 the defender's shields are revealed in FULL on every
        resolution, so driving harnesses should call ``observe_shields`` (which
        pins the exact value). This lower-bound form exists for legacy callers.
        """
        if attacks <= 0:
            return
        self.model.observe_our_attack(attacks, blocked)

    # ----------------------------------------------------------------- choice
    def choose(self, state) -> Allocation:
        state = _masked(state)
        worlds = self.model.worlds() or (WorldProxy(0, 0, 1.0),)
        team_a = _team_of(state.player)
        team_b = _team_of(state.opponent)
        roots = [(_encode_state(state, w.bank, w.shields), w.probability)
                 for w in worlds]
        mt = _cote_cfr.MicroTree(team_a, team_b, roots, depth=self.depth,
                                 cap=self.cap, start_turn=state.turn)
        override = []
        if self.value_leaf is not None:
            leaves = mt.leaf_states()
            r_opp = worlds[0].shields + worlds[0].bank
            belief_vec = [0.0] * (r_opp + 1)
            for w in worlds:
                belief_vec[w.shields] += w.probability
            override = self.value_leaf.override(team_a, team_b, leaves,
                                                belief_vec, bot_side=0)
        mt.solve(self.iters, self.gamma, override)
        actions, probs, value = mt.strategy()
        idx = _pick(probs, self.temperature, self.rng)
        a, d, b, sw = actions[idx]
        self.last_report = {
            "value": value,
            "root_actions": [(tuple(act), round(p, 4))
                             for act, p in zip(actions, probs) if p > 0.005],
            "belief": [(w.shields, w.bank, round(w.probability, 3))
                       for w in worlds],
        }
        return Allocation(attacks=a, defends=d, bonuses=b,
                          switch_to=sw if sw >= 0 else None)


class WorldProxy:
    """Fallback when the belief model has no candidates yet."""

    __slots__ = ("shields", "bank", "probability")

    def __init__(self, shields, bank, probability):
        self.shields = shields
        self.bank = bank
        self.probability = probability


def _masked(state):
    from dataclasses import replace
    from .rules import GameState
    return GameState(state.player,
                     replace(state.opponent, shields=0, bonus=0),
                     state.turn, state.player_to_move)


def _pick(probs, temperature, rng):
    if temperature > 0 and rng is not None:
        pick = rng.random() * sum(probs)
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if pick <= acc:
                return i
        return len(probs) - 1
    best, best_i = -1.0, 0
    for i, p in enumerate(probs):
        if p > best:
            best, best_i = p, i
    return best_i
