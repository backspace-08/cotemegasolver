"""Small deterministic benchmark and replay helpers for the new engine."""

import json
import random
from dataclasses import replace

from .agent import Planner
from .observation import observe
from .rules import (MAX_ACTIONS, MAX_BONUS, GameState, Type, apply,
                     attacks_to_kill, base_budget, exchange_damage,
                     legal_allocations, initial)


def _policy_state(state: GameState) -> GameState:
    """Give a human policy its own-side perspective without enemy shields."""
    return GameState(
        replace(state.player, shields=state.player.shields),
        replace(state.opponent, shields=0),
        state.turn,
        True,
    )


def choose_human_policy(state: GameState, policy: str, rng: random.Random):
    """Choose a deterministic human-like move from public own-side state."""
    view = _policy_state(state)
    moves = legal_allocations(view.player)
    if policy == "random":
        return rng.choice(moves)
    target = view.opponent.active_character

    def score(move):
        attacker = view.player.active_character
        if move.switch:
            attacker = view.player.characters[move.switch_to]
        damage = exchange_damage(attacker, target, move.attacks)
        lethal = int(damage >= target.hp)
        survival = move.defends * 700
        bonus = move.bonuses * (500 if policy == "bonus_shield" else 150)
        if policy == "greedy":
            return lethal * 100000 + damage * 10 + survival + bonus
        if policy == "bonus_shield":
            return lethal * 100000 + survival + bonus + damage * 2
        return damage

    return max(moves, key=score)


def reader_human_move(state: GameState, planner: Planner,
                      rng: random.Random):
    """Human 'reader' that predicts the bot's next shields to exploit it.

    It runs the bot's own planner forward to anticipate the bot's defensive
    shields, then commits attacks/defends knowing those shields. A deterministic
    bot (temp=0) is predicted exactly and exploited; a mixed bot (temp>0) is
    only predicted as a draw from its distribution, so the reader misfires a
    share of the time. The win-rate gap between the two is the measurable value
    of mixing. (planner.choose is read-only on history/model.)
    """
    moves = legal_allocations(state.player)
    bot_view = state.__class__(state.opponent, state.player, state.turn, True)
    pred = planner.choose(bot_view)
    pred_shields = pred.defends
    target = state.opponent.active_character

    def score(move):
        attacker = (state.player.active_character if not move.switch
                    else state.player.characters[move.switch_to])
        landed = max(0, move.attacks - pred_shields)
        dmg = exchange_damage(attacker, target, landed)
        lethal = int(dmg >= target.hp)
        return (lethal * 100000 + dmg * 10 + move.defends * 700
                + move.bonuses * 150)

    return max(moves, key=score)


def burster_human_move(state: GameState, planner: Planner,
                       rng: random.Random):
    """Human that plays the measured human exploit line: bank, then burst.

    This encodes the strategy a strong human actually used to beat the bot:

    1. If a lethal burst is available right now against the *worst* case
       (the bot holding its maximum plausible shields, i.e. its full budget),
       fire everything.
    2. Otherwise, if the bot is walling (recent turns show it holding shields
       and not attacking), bank instead of feeding attacks into shields, until
       the accumulated budget can overwhelm the wall.
    3. Otherwise trade normally, and keep enough shields to survive the bot's
       worst-case burst.

    It uses only public information: the bot's revealed shield counts from
    resolutions (``planner.history``), its own budget, and public stats. It
    never reads ``state.opponent.shields`` or ``state.opponent.bonus``.
    """
    view = _policy_state(state)
    moves = legal_allocations(view.player)
    target = view.opponent.active_character
    budget = view.player.actions

    # Public read of the bot's recent shield behaviour. planner.history holds
    # the bot's view of the HUMAN, so use the recorded resolutions of the bot's
    # shields that the harness revealed: approximate with the wall assumption.
    wall = min(MAX_ACTIONS, base_budget(state.turn))

    def score(move):
        attacker = (view.player.active_character if not move.switch
                    else view.player.characters[move.switch_to])
        # Worst case: every shield the bot could plausibly be holding.
        landed_worst = max(0, move.attacks - wall)
        dmg_worst = exchange_damage(attacker, target, landed_worst)
        kills_through_wall = int(dmg_worst >= target.hp)
        # Banking is valuable exactly when it converts into a future burst.
        bank_value = move.bonuses * 900 if not kills_through_wall else 0
        return (kills_through_wall * 1000000
                + dmg_worst * 10
                + bank_value
                + move.defends * 200)

    return max(moves, key=score)


def burster2_human_move(state: GameState, planner: Planner,
                        rng: random.Random):
    """Human line as specified by the DeepSeek test subject.

    1. Kill when the HP math says so, plus one extra attack, because the bot
       is observed to hold 1-2 shields it never revealed.
    2. When our budget cannot break a full wall, bank instead of feeding
       attacks into shields -- but spend shields to absorb the incoming burst
       rather than banking naked.
    3. Once the budget exceeds the wall, dump everything into attacks.

    Public information only: own budget, public turn, opponent HP/type.
    ``_policy_state`` already zeroes the opponent's shields.
    """
    view = _policy_state(state)
    moves = legal_allocations(view.player)
    target = view.opponent.active_character
    budget = view.player.actions
    wall = MAX_BONUS                      # bot can hold at most 4 per turn
    incoming = min(MAX_ACTIONS, base_budget(state.turn + 1) + MAX_BONUS)

    def score(move):
        attacker = (view.player.active_character if not move.switch
                    else view.player.characters[move.switch_to])
        # (1) lethal with a one-attack margin against hidden shields
        need = attacks_to_kill(attacker, target, 0)
        if need is not None and move.attacks >= need + 1:
            return 10_000_000 + move.attacks
        # (3) budget already beats the wall -> maximum volume
        if budget > wall:
            return 1_000_000 + exchange_damage(
                attacker, target, max(0, move.attacks - wall)) * 10
        # (2) cannot break the wall -> bank, but shield the incoming burst
        survives = exchange_damage(view.opponent.active_character, attacker,
                                   max(0, incoming - move.defends))
        safe = int(survives < attacker.hp)
        return safe * 100_000 + move.bonuses * 1_000 + move.defends * 100

    return max(moves, key=score)


def burster3_human_move(state: GameState, planner: Planner,
                        rng: random.Random):
    """Human line as specified by the Terra test subject.

    Terra's verdict was continuous pressure, not patient banking:

    1. Attack on nearly every turn, even at a small budget. Chip damage forces
       the bot to react instead of letting it accumulate freely.
    2. Switch when the target body has a type advantage over the bot's active,
       or to save a body that dies to the bot's next worst-case burst.
    3. When the bot's shields are low or absent, spend the whole budget on
       attacks. Do not pre-shield against a threat that is not there -- that
       passivity is exactly what sank ``burster2`` (0/40).
    4. Refinement from the DeepSeek verdict: when going for a kill, prefer one
       attack more than the HP math demands, because the bot is observed to
       hold shields it never revealed.

    The bot's shields are estimated by running the bot's own policy forward,
    the same device ``reader_human_move`` already uses: it consumes public
    information only and never reads ``state.opponent.shields``.
    """
    view = _policy_state(state)
    moves = legal_allocations(view.player)
    target = view.opponent.active_character
    threat = view.opponent.active_character
    bot_view = state.__class__(state.opponent, state.player, state.turn, True)
    predicted_shields = planner.choose(bot_view).defends
    # Worst case the bot can pay for next turn: base budget plus a full bank.
    incoming = min(MAX_ACTIONS, base_budget(state.turn + 1) + MAX_BONUS)

    def score(move):
        current = view.player.active_character
        attacker = (current if not move.switch
                    else view.player.characters[move.switch_to])
        landed = max(0, move.attacks - predicted_shields)
        damage = exchange_damage(attacker, target, landed)
        needed = attacks_to_kill(attacker, target, predicted_shields)
        kills = int(needed is not None and move.attacks >= needed)
        # (4) one extra attack over the arithmetic, for hidden shields.
        margin = int(needed is not None and move.attacks >= needed + 1)
        # (2) switch valuation: type advantage, or rescuing a doomed body.
        switch_bonus = 0
        if move.switch:
            if (exchange_damage(attacker, target, 1)
                    > exchange_damage(current, target, 1)):
                switch_bonus += 2000
            doomed = exchange_damage(threat, current, incoming) >= current.hp
            rescued = exchange_damage(threat, attacker, incoming) < attacker.hp
            if doomed and rescued:
                switch_bonus += 3000
            switch_bonus -= 800            # the action a switch costs
        # (1)+(3) continuous pressure: attacking is the default, idling is not.
        pressure = move.attacks * 400
        if move.attacks == 0:
            pressure -= 2500
        return (kills * 1_000_000 + margin * 200_000 + damage * 10
                + pressure + switch_bonus + move.defends * 50)

    return max(moves, key=score)


def run_match(seed=0, policy="greedy", depth=2, max_half_turns=100,
              ai_starts=False, temperature=0.0):
    """Run one AI-vs-human-policy match with public-information updates."""
    rng = random.Random(seed)
    types = list(Type)
    state = initial(tuple(rng.choice(types) for _ in range(3)),
                    tuple(rng.choice(types) for _ in range(3)), rng=rng)
    if ai_starts:
        state = replace(state, player_to_move=False).prepare()
    bot_rng = random.Random(seed * 1000003 + 7)
    planner = Planner(depth=depth, temperature=temperature, rng=bot_rng)
    metrics = {"missed_guaranteed_lethal": 0, "guaranteed_loss_moves": 0,
               "ai_turns": 0, "human_turns": 0, "hit_turn_limit": False}
    replay = []
    hit_limit = True
    for _ in range(max_half_turns):
        if state.player.lost or state.opponent.lost:
            hit_limit = False
            break
        if state.player_to_move:
            # Human's turn: record what the human did so the planner can
            # update its belief model of the human.  No shield observation
            # here — the human attacking reveals the AI's shields, not the
            # human's; the planner models the human, so we only call
            # observe_shields when the AI attacks and the human's shields
            # are publicly consumed.
            before = state
            if policy == "reader":
                move = reader_human_move(state, planner, rng)
            elif policy == "burster":
                move = burster_human_move(state, planner, rng)
            elif policy == "burster2":
                move = burster2_human_move(state, planner, rng)
            elif policy == "burster3":
                move = burster3_human_move(state, planner, rng)
            else:
                move = choose_human_policy(state, policy, rng)
            metrics["human_turns"] += 1
            planner.observe(move.attacks, move.bonuses, move.switch,
                            budget=before.player.actions, turn=before.turn)
        else:
            # AI's turn: state.player = human, state.opponent = AI.
            # Flip perspective so the planner sees itself as player.
            before = state
            planning_state = state.__class__(
                state.opponent, state.player, state.turn, True)
            move = planner.choose(planning_state)
            metrics["ai_turns"] += 1
            tactical = planner.last_report.get("tactical_outcome", {})
            if any(item.get("guaranteed_lethal")
                   for item in [tactical] if item) and not move.attacks:
                metrics["missed_guaranteed_lethal"] += 1
            if tactical.get("guaranteed_immediate_loss", False):
                metrics["guaranteed_loss_moves"] += 1
            # The human's shields are revealed on every resolution, attack or not
            # (before.player is the human). The planner models the human, so it
            # observes their shields symmetrically with the human's UI.
            planner.observe_shields(before.player.shields)
        replay.append({"turn": state.turn, "player_to_move": state.player_to_move,
                       "move": move.label})
        state = apply(state, move)
    metrics["hit_turn_limit"] = hit_limit
    # ``state.player`` is the human policy and ``state.opponent`` is the AI.
    # A side that ``lost`` has no living characters, so the *other* side won.
    winner = ("human" if state.opponent.lost
              else "ai" if state.player.lost else "draw")
    return {"seed": seed, "policy": policy, "depth": depth,
            "ai_starts": ai_starts,
            "winner": winner, "replay": replay, "metrics": metrics,
            "state": state}


def _seat_stats(matches):
    """Return {games, wins, losses, draws} for a list of match results."""
    return {
        "games": len(matches),
        "wins": sum(m["winner"] == "ai" for m in matches),
        "losses": sum(m["winner"] == "human" for m in matches),
        "draws": sum(m["winner"] == "draw" for m in matches),
    }


def _run_match_job(job):
    """Top-level worker for multiprocessing (picklable on Windows spawn)."""
    seed, policy, depth, max_half_turns, ai_starts, temperature = job
    return run_match(seed=seed, policy=policy, depth=depth,
                     max_half_turns=max_half_turns, ai_starts=ai_starts,
                     temperature=temperature)


def benchmark_policies(seeds=range(20), depth=2, max_half_turns=100,
                       temperature=0.0, workers=None):
    """Compare the planner against each bundled human-like policy.

    Matches are independent and seeded, so they run in parallel with
    ``workers=N`` (number of processes, e.g. cores). Results are identical to
    the sequential run; only the wall time changes. On Windows, invoke through
    a script with an ``if __name__ == "__main__"`` guard, not ``python -c``.
    """
    jobs = [
        (seed, policy, depth, max_half_turns, ai_starts, temperature)
        for seed in seeds
        for policy in ("random", "greedy", "bonus_shield")
        for ai_starts in (False, True)
    ]
    if workers and workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            matches = pool.map(_run_match_job, jobs)
    else:
        matches = [_run_match_job(job) for job in jobs]
    result = {}
    for policy in ("random", "greedy", "bonus_shield"):
        pm = [m for m in matches if m["policy"] == policy]
        ai_first_matches = [m for m in pm if m["ai_starts"]]
        human_first_matches = [m for m in pm if not m["ai_starts"]]
        combined = _seat_stats(pm)
        result[policy] = {
            # Combined totals (kept for backward compatibility)
            "games": combined["games"],
            "wins": combined["wins"],
            "losses": combined["losses"],
            "draws": combined["draws"],
            # Per-seat breakdown
            "ai_first": _seat_stats(ai_first_matches),
            "human_first": _seat_stats(human_first_matches),
            "missed_guaranteed_lethal": sum(
                m["metrics"]["missed_guaranteed_lethal"] for m in pm),
            "matches": pm,
        }
    return result


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
    rng = random.Random(seed)
    state = initial(tuple(rng.choice(types) for _ in range(3)),
                    tuple(rng.choice(types) for _ in range(3)), rng=rng)
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
            "move_quality": planner.last_report.get("move_quality", {}),
        })
        target = state.opponent if state.player_to_move else state.player
        planner.observe_shields(target.shields)
        planners[not state.player_to_move].observe(
            move.attacks, move.bonuses, move.switch,
            budget=(state.player if state.player_to_move
                    else state.opponent).actions, turn=state.turn)
        state = apply(state, move)
    winner = "opponent" if state.player.lost else "player" if state.opponent.lost else "draw"
    return {"seed": seed, "winner": winner, "replay": replay, "state": state}


def write_replay(report, path):
    serializable = {key: value for key, value in report.items() if key != "state"}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, default=str)
