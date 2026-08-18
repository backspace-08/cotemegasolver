"""Archetype heuristic agents + baseline benchmark vs the Planner.

Four pure play styles (sensible heuristics, not idiot bots - each uses a belief
about the opponent's shields, finishes low-HP actives, and spends the whole
budget):

  Aggressor (tempo)     - attacks most turns, banks to break shield walls.
  Cautious  (counter)   - shields a moderate amount, then counter-attacks.
  Banker    (economy)   - banks to a big budget, then bursts through shields.
  Switcher  (type)      - switches to the best type matchup, then attacks.

Benchmark: python server/archetypes.py --games 40 [--workers N]
Each archetype is matched against the Planner (both seats), same seed split as
cfr_vs_planner, and the winrate / draw-rate / half-turns baseline is printed.
"""
import argparse
import os
import random
import sys
import time
from dataclasses import replace
from math import ceil
from multiprocessing import Pool

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

from cote_megaverse.rules import (Allocation, GameState, MAX_ACTIONS, Type,  # noqa: E402
                                  apply, attacks_to_kill, base_budget, initial,
                                  multiplier, per_hit_damage)
from cote_megaverse.infoset import OpponentModel  # noqa: E402
from cote_megaverse.agent import Planner  # noqa: E402


class Archetype:
    """Base: public-info belief about the opponent's shields + the bot interface."""

    name = "archetype"

    def __init__(self, rng=None):
        self.model = OpponentModel()
        self.rng = rng or random.Random()

    def observe(self, attacks, bonuses=None, switched=False, budget=None, *,
                turn):
        if budget is None:
            budget = attacks + int(switched)
        self.model.observe_turn(turn, budget, attacks, switched)

    def observe_shields(self, shields):
        self.model.observe_our_attack(shields + 1, shields)

    def _exp_shields(self):
        d = self.model.shield_distribution()
        if not d:
            return 0.0
        return sum(s * p for s, p in d.items())

    def choose(self, state):
        raise NotImplementedError


class Aggressor(Archetype):
    """Tempo: attack as often as possible; bank to break a shield wall; shield a
    little only when otherwise doomed."""

    name = "Aggressor"

    def choose(self, state):
        B = state.player.actions
        atk_c = state.player.active_character
        def_c = state.opponent.active_character
        sh = round(self._exp_shields())
        need = attacks_to_kill(atk_c, def_c, sh, B)
        if need is not None and need <= B:
            return _spend(B, need)
        if sh >= B:  # the wall is thick -> bank toward a burst (with a token shield)
            return _spend(B, 0, d=min(2, B))
        opp_ph = per_hit_damage(def_c, atk_c)
        opp_next = min(MAX_ACTIONS, base_budget(state.turn + 1) + 4)
        if opp_ph > 0 and opp_ph * opp_next >= atk_c.hp and B >= 2:
            return _spend(B, B - 1, d=1)  # keep a shield to survive
        return _spend(B, B)


class Cautious(Archetype):
    """Counter-attack: shield a moderate amount to absorb the retaliation, then
    attack with the rest; kill the active when it is lethal."""

    name = "Cautious"

    def choose(self, state):
        B = state.player.actions
        atk_c = state.player.active_character
        def_c = state.opponent.active_character
        sh = round(self._exp_shields())
        need = attacks_to_kill(atk_c, def_c, sh, B)
        if need is not None and need <= B:
            return _spend(B, need)
        opp_ph = per_hit_damage(def_c, atk_c)
        opp_next = min(MAX_ACTIONS, base_budget(state.turn + 1) + 2)
        threat = 0
        if opp_ph > 0:
            threat = min(opp_next, ceil(atk_c.hp / opp_ph))
        d = min(B, threat) if threat > 0 else 0
        d = min(d, (B // 2) + 1)  # never max-shield; keep some attack
        return _spend(B, B - d, d=d)


class Banker(Archetype):
    """Economy: bank to a big budget, then burst through shields / kill."""

    name = "Banker"

    def choose(self, state):
        B = state.player.actions
        if B >= 6:  # burst
            atk_c = state.player.active_character
            def_c = state.opponent.active_character
            sh = round(self._exp_shields())
            need = attacks_to_kill(atk_c, def_c, sh, B)
            if need is not None and need <= B:
                return _spend(B, need)
            return _spend(B, B)
        return _spend(B, 0, d=min(2, B))  # a token shield while banking


class Switcher(Archetype):
    """Type: switch to the best matchup vs the opponent's active, then attack."""

    name = "Switcher"

    def choose(self, state):
        B = state.player.actions
        def_c = state.opponent.active_character
        cur = state.player.active_character
        cur_val = multiplier(cur.type, def_c.type) * cur.atk
        best_i, best_val = None, cur_val
        for i, c in enumerate(state.player.characters):
            if i == state.player.active or not c.alive:
                continue
            v = multiplier(c.type, def_c.type) * c.atk
            if v > best_val:
                best_i, best_val = i, v
        sh = round(self._exp_shields())
        if best_i is not None and B >= 2:
            rem = B - 1
            atk_c = state.player.characters[best_i]
            need = attacks_to_kill(atk_c, def_c, sh, rem)
            if need is not None and need <= rem:
                return _spend(B, need, sw=best_i)
            return _spend(B, rem, sw=best_i)
        need = attacks_to_kill(cur, def_c, sh, B)
        if need is not None and need <= B:
            return _spend(B, need)
        return _spend(B, B)


ARCHETYPES = {"aggressor": Aggressor, "cautious": Cautious,
              "banker": Banker, "switcher": Switcher}


def _spend(B, a, d=0, sw=None):
    """Build a legal allocation (a+d+b = B, bank capped at 4) filling the
    leftover budget with bank, then shields."""
    rem = B - a - d - (1 if sw is not None else 0)
    b = min(rem, 4)
    d += rem - b
    return Allocation(a, d, b, sw)


def run_match(cls, seed, new_first, pl_depth=2, pl_max_nodes=2000):
    rng = random.Random(seed)
    types = list(Type)
    state = initial(tuple(rng.choice(types) for _ in range(3)),
                    tuple(rng.choice(types) for _ in range(3)), rng=rng)
    if not new_first:
        state = replace(state, player_to_move=False).prepare()
    agent = cls(rng=random.Random(seed * 1000003 + 17))
    pl = Planner(depth=pl_depth, max_nodes=pl_max_nodes)
    lat = []
    for _ in range(80):
        if state.player.lost or state.opponent.lost:
            break
        if state.player_to_move:  # archetype agent
            before = state
            planning = GameState(state.player, state.opponent, state.turn, True)
            t0 = time.time()
            move = agent.choose(planning)
            lat.append((time.time() - t0) * 1000)
            state = apply(state, move)
            agent.observe_shields(before.opponent.shields)
            pl.observe(move.attacks, move.bonuses, move.switch,
                       budget=before.player.actions, turn=before.turn)
        else:  # planner
            before = state
            planning = GameState(state.opponent, state.player, state.turn, True)
            move = pl.choose(planning)
            state = apply(state, move)
            pl.observe_shields(before.player.shields)
            agent.observe(move.attacks, move.bonuses, move.switch,
                          budget=before.opponent.actions, turn=before.turn)
    if state.opponent.lost:
        winner = "AGENT"
    elif state.player.lost:
        winner = "PL"
    else:
        winner = "DRAW"
    return {"winner": winner, "half_turns": len(lat),
            "lat_ms": sum(lat) / len(lat) if lat else 0}


def worker(task):
    name, seed, nf, pd, pn = task
    cls = ARCHETYPES[name]
    return {"name": name, "seed": seed, "new_first": nf,
            **run_match(cls, seed, nf, pd, pn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--pl-depth", type=int, default=2)
    ap.add_argument("--pl-max-nodes", type=int, default=2000)
    a = ap.parse_args()
    tasks = []
    for name in ARCHETYPES:
        for i in range(a.games // 2):
            seed = 1100 + i
            for seat in (True, False):
                tasks.append((name, seed, seat, a.pl_depth, a.pl_max_nodes))
    with Pool(a.workers) as pool:
        results = list(pool.imap_unordered(worker, tasks))
    for name, cls in ARCHETYPES.items():
        rs = [r for r in results if r["name"] == name]
        n = len(rs)
        w = sum(1 for r in rs if r["winner"] == "AGENT")
        d = sum(1 for r in rs if r["winner"] == "DRAW")
        ht = [r["half_turns"] for r in rs]
        lat = [r["lat_ms"] for r in rs]
        print(f"{name:10s} vs Planner: AGENT={w} PL={n-w-d} DRAW={d} "
              f"winrate={100.0*w/n:.1f}%  avg_half={sum(ht)/n:.1f} "
              f"lat={sum(lat)/len(lat):.0f}ms")


if __name__ == "__main__":
    main()
