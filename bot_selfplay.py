import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from multiprocessing import Pool
import random as _random
from cote_megaverse.agent import Planner
from cote_megaverse.rules import initial, Type, apply

def play(args):
    seed, first_is_player = args
    rng = _random.Random(seed)
    types = list(Type)
    state = initial(tuple(rng.choice(types) for _ in range(3)),
                    tuple(rng.choice(types) for _ in range(3)), rng=rng)
    if not first_is_player:
        from dataclasses import replace
        state = replace(state, player_to_move=False).prepare()
    first_p2m = state.player_to_move
    pl = {True: Planner(depth=1), False: Planner(depth=1)}
    turns = 0
    while not (state.player.lost or state.opponent.lost) and turns < 100:
        p = pl[state.player_to_move]
        planning = state if state.player_to_move else state.__class__(
            state.opponent, state.player, state.turn, True)
        move = p.choose(planning)
        target = state.opponent if state.player_to_move else state.player
        p.observe_shields(target.shields)
        pl[not state.player_to_move].observe(
            move.attacks, move.bonuses, move.switch,
            budget=(state.player if state.player_to_move
                    else state.opponent).actions, turn=state.turn)
        state = apply(state, move)
        turns += 1
    if first_p2m:
        if state.opponent.lost:
            return "first"
        if state.player.lost:
            return "second"
    else:
        if state.player.lost:
            return "first"
        if state.opponent.lost:
            return "second"
    return "draw"

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    jobs = [(s, f) for f in (True, False) for s in range(n)]
    t0 = time.time()
    with Pool(workers) as pool:
        res = list(pool.imap_unordered(play, jobs))
    f = res.count("first"); s = res.count("second"); d = res.count("draw")
    print("bot self-play (depth1): first=%d second=%d draw=%d  (first %.1f%% of decided, %.0fs)"
          % (f, s, d, 100.0 * f / (f + s) if f + s else 0, time.time() - t0), flush=True)
