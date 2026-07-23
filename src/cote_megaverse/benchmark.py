"""Small deterministic benchmark and replay helpers for the new engine."""

import json
import random
from dataclasses import asdict

from .agent import Planner
from .observation import observe
from .rules import GameState, Type, apply, initial


def analyze_position(state: GameState, depth=3) -> dict:
    planner = Planner(depth=depth)
    move = planner.choose(state)
    return {
        "turn": state.turn,
        "player_to_move": state.player_to_move,
        "observation": observe(state),
        "report": planner.last_report,
        "best_move": move.label,
    }


def run_self_play(seed=0, max_half_turns=40, depth=2):
    """Run two independent fair planners and return a replayable report."""
    random.seed(seed)
    types = list(Type)
    state = initial(tuple(random.choice(types) for _ in range(3)),
                    tuple(random.choice(types) for _ in range(3)))
    planners = {True: Planner(depth=depth), False: Planner(depth=depth)}
    replay = []
    for _ in range(max_half_turns):
        if state.player.lost or state.opponent.lost:
            break
        planner = planners[state.player_to_move]
        planning_state = state if state.player_to_move else state.__class__(
            state.opponent, state.player, state.turn, True)
        move = planner.choose(planning_state)
        replay.append({
            "turn": state.turn,
            "player_to_move": state.player_to_move,
            "move": move.label,
            "report": planner.last_report,
        })
        if state.player_to_move:
            state = apply(state, move)
        else:
            child = apply(planning_state, move)
            state = child.__class__(child.opponent, child.player, child.turn, False)
    winner = "opponent" if state.player.lost else "player" if state.opponent.lost else "draw"
    return {"seed": seed, "winner": winner, "replay": replay, "state": state}


def write_replay(report, path):
    serializable = {key: value for key, value in report.items() if key != "state"}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, default=str)
