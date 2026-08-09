# PROJECT MEMORY — COTE Megaverse

This file exists so that a future agent (after context compaction, or a fresh
session) can pick up the project without re-deriving everything. It captures
what the game is, how the AI is built, the key decisions and *why*, the
measured results, the known weaknesses, and where to continue.

The normative spec lives in `AGENT.md` (read it for exact rules). This file is
the project's institutional memory.

---

## 1. The project in one paragraph

Build an AI that beats a human at a turn-based, hidden-intent combat game
(COTE Megaverse) with the highest achievable win rate. The game is small and
structurally simple, so the AI is a **hand-built belief-based planner**, not a
learned network. The core challenge is imperfect information: when a side acts,
the opponent's shields and stored bonus are hidden. The AI wins by playing
soundly (punishing human errors) while not being trivially readable. The game
itself is a great benchmark: an honest external arbiter ("BOT WIN" after ~12
turns) that cannot be argued with.

## 2. Game rules (compact)

- Two sides, 3 characters each. Type cycle `A > B > C > D > A` (attack the type
  you beat → x1.3, the type that beats you → x0.7, else x1.0). HP 5700-6300,
  ATK 1900-2100. Win by killing all three.
- Each turn: budget = base(turn) + bank, cap 8 (base: turn1=1, 2-4=2, 5-6=3,
  7+=4). Spend EVERY action on attacks / defends / bonuses / at most one switch
  (costs 1).
- Defends become shields: each blocks 1 attack on the opponent's NEXT turn,
  then expires. Damage = aggregate, rounded once to hundreds.
- A kill forces a free promotion; the dead character is removed from the stack
  entirely.
- Draw = hitting the configured turn limit.

## 3. The information model (the heart)

- Public: types, HP, ATK, actives, stacks, turn, acting budget, resolved
  results, visible switches.
- Hidden at decision time: the opponent's current shields and stored bank
  (their last remainder's split into defends vs bank).
- Three leaks at three delays: attacks reveal instantly; bank reveals when
  spent (turn N banked → visible turn N+2 via budget); shields reveal on every
  resolution (whether hit or expired). A wasted shield is always visible.
- `remainder = budget - attacks - switch`; every split `(shields, bank)` of it
  is a live world. A remainder of zero is the one exact case.
- **Fairness rule**: the bot must never read the opponent's resolver fields
  (`shields`, `bonus`); belief comes only from `OpponentModel`. The harness and
  benchmark enforce the same boundary. Historical bug: the benchmark fed the
  bot's OWN shields into its model of the human, and the CLI printed the bot's
  full allocation — both leaks made old measurements invalid.

## 4. Architecture

- `rules.py` — immutable state, legal moves, damage, transitions. Single
  authority.
- `infoset.py` — `OpponentModel`: joint `(shields, bank)` belief. Prior is
  **binomial** (each remainder action → shield with prob `defend_share`), so
  evidence moves belief sharply; attack observations feed the behavioural
  prior. EPSILON keeps every world alive.
- `agent.py` — `Planner`. Four layers:
  1. Belief (from OpponentModel).
  2. **Gates** (hard): match-win first; exclude moves that lose in any joint
     world (`_reply_kills_us` sizes the reply from the *believed* bank —
     masking it to zero would blind the gate to bursts); dominated switches
     never sampled.
  3. **Scoring** (safe candidates): material, survival, tempo, switch value,
     burst setup, and strategic terms — `punish_banking`, `deny_burst`
     (calibrate shields vs the worst-case burst; breaks the "turtle"),
     `desperation` (threat when behind), `burst_setup`.
  4. **Context-conditional mixing**: deterministic by default (temp=0); with
     temp>0 samples only among moves within `band_fraction` of the best, only
     when ≥2 such moves exist.
- `strategy.py` — `Objective` classifier (advisory) and `switch_value`.
- `observation.py` — public boundary (hides opponent bank/shields).
- `benchmark.py` — seeded self-play + exploit policies `reader` and `burster`.
- `solve1v1.py` — exact 1v1 CFR-style solver (laboratory, not converged).
- `play.py` — harness (see §7), `track_winrate.py` — stats with Wilson CI.

## 5. Key decisions and WHY (the reasoning)

- **Deterministic base > heavy mixing.** Exploiting determinism costs the
  opponent "compute the exact move" — humans don't. Measured: temp=0 beats
  random 100%, temp=0.3 83%. Mixing is only surgical (bluff spots), never
  global.
- **Gates before scores.** A certain kill must be taken regardless of score; a
  move that loses in any world is excluded when a safe alternative exists.
  Scores are tie-breaks. Lesson from a near-bug: a suicidal kill (reply kills
  you) is correctly excluded by the gate — don't tune scores to force it.
- **Shields revealed every resolution** (not just on attack). The opponent sees
  wasted shields; the bot sees them too. This is fair (a human accumulates
  this history) and enables reading patterns via the harness's history panel.
- **Binomial prior** replaced a flat linear prior that barely moved with
  evidence — a non-shielder got ~9% weight on "holds 4 shields", so the bot
  couldn't punish banking. Now repeated "0 shields" observations sharpen the
  prior.
- **Joint (shields, bank) worlds** in the loss gate — the bank sets the reply
  budget. The bank-and-burst exploit (bank 4-6 turns, burst 8 through shields)
  was the main way strong models beat the old bot.
- **The game's ceiling vs a perfect opponent is ~50%** (equilibrium). Win rate
  above that comes from punishing human errors, not from mixing. Mixing only
  prevents being read.

## 6. Measured results

- Test suite: **48/48 green**.
- Vs bundled policies (100 seeds, depth 2): random **195-4-1**, greedy/bonus
  **185-0-15**, 0 missed lethals. ~458s on 16 cores (was ~40 min sequential).
- Vs strong subagent (Opus, fair info, same 4 seeds): bot **3-1** (was 1-3
  before the hardening).
- Vs weak subagent: ~9-1.
- User's session (BEFORE hardening): 55% user win rate (seat split: user-first
  67%, bot-first 45% — first-mover advantage ~20 points).

## 7. Tooling

- `play.py new/move/view/end/session --run <name> --temp T`: headless harness.
  Random first mover; budget auto-banks the leftover (no summing); 1-based
  switches; shields revealed every resolution; history panel; per-run
  isolation under `runs/<name>/`. Interactive session for a human: `session`.
- `track_winrate.py add/show/reset --run <name>`: Wilson 95% CI, per-seat and
  per-block stats, stop advice. ~100 games → ±10%, ~200 → ±7%.
- `run_baseline.py --seeds N --limit L --workers W --out f`: parallel
  benchmark (Windows needs the file-based runner, NOT `python -c`).
- Subagents in `.opencode/agent/` (flagships `opus5-flagman`,
  `sol56-flagman`; free models) play as the human; results land in their
  `runs/<name>/`.

## 8. Known weaknesses / where to improve next

1. **Re-measure vs the user** — the 55% figure predates all the hardening. A
   fresh 20-game session is the single most valuable next step.
2. **Midgame "head-on" attacks without shields** (found by Opus) — the bot
   sometimes over-commits to offense, giving free trades. Targeted fix
   (attack but keep minimal shields vs a strong counter) must not reintroduce
   the turtle.
3. **15 draws** vs greedy/bonus — the bot doesn't close out stalls; be more
   aggressive at finishing when the opponent is passive.
4. **4 random losses** — check if bot errors or unlucky stat draws.
5. **Principled evaluation** — the scoring terms are hand-tuned and
   overlapping; a proper utility (calibrated vs solve1v1) is the deferred big
   step for midgame decisions.
6. **Burst telegraphing** — the bank is structurally visible; the bot is
   predictable right before a burst and exposed right after.

## 9. History of fixed bugs (do not regress)

- Benchmark winner label inverted (`opponent` is the AI). Old numbers void.
- Forced promotion used the actor's order and kept dead/duplicate stack
  entries.
- Switch-target indexing 0-based vs 1-based mismatch.
- Shields only revealed on attack (now every resolution).
- Harness/CLI leaked the bot's full allocation and the opponent's bank.
- Flat belief prior didn't learn.

## 10. Workflow notes

- Verify with `python -m unittest discover -s tests -p "test_*.py"` (48 tests).
- After planner changes, re-run `run_baseline.py --seeds 100 --limit 100
  --workers 16` and check vs the exploit policies (`reader`, `burster`) that a
  hole stays closed.
- Commit cleanly (`.gitignore` ignores `runs/`, `*.pkl`, benchmark outputs).
- Do not reintroduce genome evolution / LSTM / Numba legacy planners /
  perfect-information behavior without a design decision.
