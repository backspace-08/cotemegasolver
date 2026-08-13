"""1v1 match harness: GTO solver policy vs the bot's Planner, W/L/D, both seats.

The solver plays the hits-abstraction game; the bot (Planner) plays its native
real-HP game. Bridge: at the bot's decision the abstracted hit counts are mapped
to representative real HP (bucket top = hits * per-hit damage), so the bot's
perceived hits-to-kill equals the solver's. All public observations (attacks,
budgets, revealed shields) are identical in both views.
"""
import random

from .agent import Planner, PublicHistory
from .rules import Character, Side, GameState, Type
from .solver1v1 import Solver


class SolverPolicy:
    """Plays the equilibrium average strategy, sampling each info set."""

    def __init__(self, solver, seed=0):
        self.solver = solver
        self.rng = random.Random(seed)

    def act(self, st):
        info = self.solver.info_key(st)
        sig = self.solver.average_strategy(info)
        acts = self.solver.actions(st)
        total = sum(sig)
        pick = self.rng.random() * total
        acc = 0.0
        for j, p in enumerate(sig):
            acc += p
            if pick <= acc:
                return acts[j]
        return acts[-1]


class BotPolicy:
    """Wraps Planner to play one character in a 1v1, observing public facts."""

    def __init__(self, solver, is_solver_side_A, bot_type, bot_atk, bot_hp,
                 seed=0, clean_history=False):
        self.solver = solver
        self.is_A = is_solver_side_A
        self.bot_type, self.bot_atk, self.bot_hp = bot_type, bot_atk, bot_hp
        self.planner = Planner(depth=2, temperature=0.0)
        # NOTE: PublicHistory is fair since the bonuses-leak fix (AGENT.md §14);
        # `clean_history` is kept only for API compatibility and is a no-op.
        self.rng = random.Random(seed)

    def _planning(self, st):
        hA, hB, bankA, bankB, shA, shB, turn, to_move = st
        if self.is_A:
            bot_hits, bot_bank, bot_sh = hA, bankA, shA
            opp_hits, opp_bank, opp_sh = hB, bankB, shB
        else:
            bot_hits, bot_bank, bot_sh = hB, bankB, shB
            opp_hits, opp_bank, opp_sh = hA, bankA, shA
        # representative real HP: bucket top keeps the bot's hits-to-kill exact
        bot_hp = self.solver.real_hp(bot_hits, "B" if self.is_A else "A")
        opp_hp = self.solver.real_hp(opp_hits, "A" if self.is_A else "B")
        dead = Character(Type.A, 0, 0, 0)
        bot_side = Side(
            (dead, dead, Character(self.bot_type, bot_hp, self.bot_atk, bot_hp)),
            active=2, stack_order=(2,), bonus=bot_bank, shields=bot_sh)
        opp_side = Side(
            (dead, dead, Character(self._opp_type(), opp_hp, self._opp_atk(),
                                   opp_hp)),
            active=2, stack_order=(2,), bonus=opp_bank, shields=opp_sh)
        return GameState(bot_side, opp_side, turn, True).prepare()

    def _opp_type(self):
        return self.solver.typeA if not self.is_A else self.solver.typeB

    def _opp_atk(self):
        return self.solver.atkA if not self.is_A else self.solver.atkB

    def act(self, st):
        planning = self._planning(st)
        move = self.planner.choose(planning)
        self.last_move = move
        return (move.attacks, move.defends, move.bonuses)

    def observe_solver_move(self, st, act):
        hA, hB, bankA, bankB, shA, shB, turn, to_move = st
        a, d, b = act
        from .solver1v1 import base as _b
        opp_budget = min(8, _b(turn) + (bankB if self.is_A else bankA))
        # Fairness: bonuses are hidden. Passing the real `b` would let the bot's
        # PublicHistory derive the opponent's exact shields (AGENT.md §14 debt).
        # Pass 0 so history derives only the public remainder.
        self.planner.observe(a, 0, False, budget=opp_budget)

    def observe_solver_shields(self, shields):
        self.planner.observe_shields(shields)


def play_match(solver, seat, seed=0, verbose=False, mode="bot",
               clean_history=False):
    """seat: 'bot_first' -> A is the bot (or solver in 'self' mode); 'opp_first'
    -> A is the solver. mode 'self' plays solver vs solver (sanity)."""
    bot_type, bot_atk, bot_hp = solver.typeB, solver.atkB, solver.hpB
    if mode == "self":
        polA = SolverPolicy(solver, seed=seed)
        polB = SolverPolicy(solver, seed=seed + 1)
        bot_pol_index = None
    elif seat == "bot_first":
        polA = BotPolicy(solver, True, bot_type, bot_atk, bot_hp, seed=seed,
                         clean_history=clean_history)
        polB = SolverPolicy(solver, seed=seed + 1)
        bot_pol_index = 0
    else:
        polA = SolverPolicy(solver, seed=seed)
        polB = BotPolicy(solver, False, bot_type, bot_atk, bot_hp, seed=seed + 1,
                         clean_history=clean_history)
        bot_pol_index = 1
    st = solver.start_states[0]
    plies = 0
    while True:
        t = solver.terminal(st)
        if t is not None:
            return t, plies
        before = st
        act = (polA if st[7] == 0 else polB).act(st)
        if verbose:
            print(f"  T{st[6]} {'A' if st[7]==0 else 'B'} a{act[0]}/d{act[1]}/b{act[2]} "
                  f"hitsA={st[0]} hitsB={st[1]}", flush=True)
        acting = 0 if st[7] == 0 else 1
        if mode == "self":
            pass  # no bot observations needed
        elif acting == bot_pol_index:
            # the bot acted; the solver's shields are revealed this resolution
            defender_sh = before[5] if acting == 0 else before[4]
            (polA if bot_pol_index == 0 else polB).observe_solver_shields(
                defender_sh)
        else:
            (polA if bot_pol_index == 0 else polB).observe_solver_move(
                before, act)
        st = solver.transition(st, act)
        plies += 1


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from cote_megaverse.rules import Type
    print("=== SMOKE: hits solver, C(10000,2000) vs C(10000,2000), turn 1 ===")
    s = Solver(Type.C, Type.C, 2000, 2000, 10000, 10000, cap=8, stall_cap=3,
               start_turn=1)
    print(f"states={len(s.states)} hits A={s.hA0} B={s.hB0}")
    s.run(80, silent=True)
    v = s.value[s.info_of[s.roots[0]]]
    e = (s.best_response_value(0) - v) + (v - s.best_response_value(1))
    print(f"V={v:+.3f} exploitability={e:.4f}")
    print("--- solver vs solver (self-play, sanity: ~0.5 each) ---")
    import time
    w = l = d = 0
    n = 100
    t0 = time.time()
    for seed in range(n):
        res, _ = play_match(s, "opp_first", seed=seed, mode="self")
        w += (res > 0); l += (res < 0); d += (res == 0)
    print(f"solver A vs solver B: W={w} L={l} D={d} "
          f"({(w + 0.5*d)/n*100:.1f}% A-side pts) {time.time()-t0:.1f}s")
    print("--- solver vs bot: one verbose game per seat ---")
    for seat in ("bot_first", "opp_first"):
        res, plies = play_match(s, seat, seed=0, verbose=True, mode="bot")
        print(f"{seat}: result A={res:+.1f} ({plies} plies)")
