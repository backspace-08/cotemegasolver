"""Test the final full-history 1v1 solver (checkpoint) against the Planner.

Both seats. The solver plays the converged equilibrium (sampling the average
strategy of its full-history info set); the Planner plays its real-HP game via
the hits bridge from match_1v1. Times a small sample first so we know how long
a full benchmark would take.
"""
import os
import pickle
import random
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))

from cote_megaverse.match_1v1 import BotPolicy  # noqa: E402
from cote_megaverse.solver_tree_fh import Solver  # noqa: E402


class FHSolverPolicy:
    """Plays the converged full-history equilibrium, tracking its OWN
    observation sequence (the full-history info sets are keyed by the acting
    player's whole observation history, not the collapsed state key)."""

    def __init__(self, solver, seed=0):
        self.solver = solver
        self.rng = random.Random(seed)
        self.seq = ()

    def act(self, st):
        ik = self.solver.info_key(st)
        self.seq = self.seq + (ik,)
        sig = self.solver.average_strategy(self.seq)
        acts = self.solver.actions(st)
        total = sum(sig)
        pick = self.rng.random() * total
        acc = 0.0
        for j, p in enumerate(sig):
            acc += p
            if pick <= acc:
                return acts[j]
        return acts[-1]


def load_solver(ckpt, params_path):
    params = pickle.load(open(params_path, "rb"))
    data = pickle.load(open(ckpt, "rb"))
    s = Solver(**params)
    s.avg = data["avg"]
    s.regret = data["regret"]
    s._t = data["_t"]
    return s


def play_match(solver, seat, seed=0, verbose=False):
    """seat: 'bot_first' -> A=Planner, B=solver; 'opp_first' -> A=solver, B=Planner."""
    bot_type, bot_atk, bot_hp = solver.typeB, solver.atkB, solver.hpB
    if seat == "bot_first":
        polA = BotPolicy(solver, True, bot_type, bot_atk, bot_hp, seed=seed)
        polB = FHSolverPolicy(solver, seed=seed + 1)
        bot_pol_index = 0
    else:
        polA = FHSolverPolicy(solver, seed=seed)
        polB = BotPolicy(solver, False, bot_type, bot_atk, bot_hp, seed=seed + 1)
        bot_pol_index = 1
    st = solver.start_states[0]
    plies = 0
    while True:
        t = solver.terminal(st)
        if t is not None:
            return t, plies
        before = st
        act = (polA if st[7] == 0 else polB).act(st)
        acting = 0 if st[7] == 0 else 1
        if acting == bot_pol_index:
            defender_sh = before[5] if acting == 0 else before[4]
            (polA if bot_pol_index == 0 else polB).observe_solver_shields(
                defender_sh)
        else:
            (polA if bot_pol_index == 0 else polB).observe_solver_move(
                before, act)
        if verbose:
            print(f"  T{st[6]} {'A' if st[7]==0 else 'B'} "
                  f"a{act[0]}/d{act[1]}/b{act[2]} hits=({st[0]},{st[1]})")
        st = solver.transition(st, act)
        plies += 1


def main():
    ckpt = os.path.join(BASE, "server_out_fh", "ckpt_fh_cap6.pkl")
    params_path = os.path.join(BASE, "server_out_fh", "params_fh_cap6.pkl")
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    seed_start = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    t0 = time.time()
    s = load_solver(ckpt, params_path)
    print(f"solver: C({s.hpA},{s.atkA}) vs C({s.hpB},{s.atkB}) cap={s.cap} "
          f"_t={s._t} nodes={len(s.states)} infos={s.n_infos} "
          f"load={time.time()-t0:.1f}s")

    w = l = d = 0
    t1 = time.time()
    per_seat = {}
    for seat in ("bot_first", "opp_first"):
        per_seat[seat] = [0, 0, 0]
    for i in range(games):
        seat = "bot_first" if i % 2 == 0 else "opp_first"
        res, plies = play_match(s, seat, seed=seed_start + i)
        if res > 0:
            w += 1
            per_seat[seat][0] += 1
        elif res < 0:
            l += 1
            per_seat[seat][1] += 1
        else:
            d += 1
            per_seat[seat][2] += 1
    dt = time.time() - t1
    n = w + l + d
    print(f"\n{games} games vs Planner: W={w} L={l} D={d}  "
          f"solver-points={(w + 0.5 * d) / n * 100:.1f}%  "
          f"time={dt:.1f}s ({dt / n:.2f}s/game)")
    for seat in ("bot_first", "opp_first"):
        sw, sl, sd = per_seat[seat]
        print(f"  {seat:11s} (Planner={'A' if seat=='bot_first' else 'B'}, "
              f"solver={'B' if seat=='bot_first' else 'A'}): "
              f"W={sw} L={sl} D={sd}")


if __name__ == "__main__":
    main()
