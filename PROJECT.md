# COTE Megaverse — Project Structure

This document explains the repository layout: what lives where, why it is
needed, how the parts connect, and how everything works together. The exact
game rules live in `RULES.md`.

---

## 1. Project goal

Build a strong AI for the turn-based game COTE Megaverse (3v3, hidden
shields/bank). The components are:

- **1v1 solver** — exact duel equilibrium values (used as 2v2/3v3 leaf values).
- **Bot (CFR)** — plays through the Rust engine, re-solves a subtree under a belief.
- **Planner** — strong heuristic opponent (benchmark).
- **Learning infra (ReBeL)** — value-net for belief-conditioned leaves.

---

## 2. Game rules

`RULES.md` is the exact rule specification: turns, action budget, bank,
shields, switch, damage, what is visible vs hidden, hidden-info reveal, match
end. It is authoritative for all simulations and harnesses.

---

## 3. Game core — `src/cote_megaverse/rules.py`

**Authoritative rules implementation.** Immutable simulator:
`GameState`/`Side`/`Character`, `apply()` (move resolution), `prepare()`
(drain bank into the budget), `initial()` (random team generation),
`exchange_damage()` / `per_hit_damage()` (damage, per-hit rounding),
`multiplier()` (type cycle), `base_budget()` (turn schedule),
`legal_allocations()`.

All simulations (game, bots, harnesses) run through `rules.py`.

**Damage model (unified on rules.py, per-hit):**
```
per_hit = round(atk * mult / 100) * 100      # Python round(), banker's half-to-even
damage  = per_hit * hits
```
Each hit is rounded once; total damage is linear in hits, so
`hits_to_kill = ceil(hp / per_hit)` is exact. The Rust engine
(`Trunk::per_hit_units` + `Trunk::hits_to_kill`) and `trunk.py` use the same
formula. The 1v1 table is built at atk=2000/mult=1.0, where
`round(2000/100)*100 = 2000` is exact — the table is damage-model independent.

## 4. Information model — `observation.py`, `infoset.py`

- `observation.py` — public observation: what one side may know. The
  opponent's hidden fields (shields/bank) never cross the boundary.
- `infoset.py` — `OpponentModel`: a belief (distribution) over the opponent's
  hidden `(shields, bank)` split, inferred only from public facts: budget,
  attacks, switch, remainder `R = budget − attacks − switch`, and reveals.

## 5. Opponents

- `agent.py` — **Planner (v3)**, the main heuristic opponent: belief model,
  tactical facts (kill thresholds, damage), depth-limited search (alpha-beta),
  strategic layers (punish banking, burst setup, survival). Benchmark.
- `agent_v2.py` — **Planner (v2)**, frozen variant; used in `botvbot.py` to
  compare generations.
- `cfr_bot.py` — **CFR bot**: re-solves a subtree through the Rust engine
  (`_cote_cfr.MicroTree` / `solve_micro_belief`); duel leaves come from the
  1v1 table, 2v2/3v3 leaves are material (or the value-net if provided).

## 6. Solvers

- `trunk.py` — **3v3 trunk** for the CFR-D / subgame solver: the whole match,
  switches, promotions, banks, shields; when reduced to 1v1 it hands the value
  to a leaf (the 1v1 table).
- `strategy.py` — switch value, marginal bank value, strategic objectives
  (`Objective`).

The old full-history 1v1 solver (`solver_tree_fh.py`) and its harnesses were
**removed** — the table `cote_cfr/1v1_table.csv` is the only 1v1 artifact.

## 7. Rust engine and the table — `cote_cfr/`

`cote_cfr/src/lib.rs` (PyO3, `import _cote_cfr`):

- `MicroTree` / `solve_micro` / `solve_micro_belief` — depth-limited CFR,
  micro-trees, belief roots, leaves.
- `solve_1v1_step` — batched backward-induction step for building the 1v1
  table (sliced, multiprocessing driver `server/build_1v1_table.py`).
- `load_1v1_table` — loads the belief table.
- `MicroSolver::root_strategy` returns probabilities from the **average**
  profile and the value from `avg_profile_value()` (a backward pass under the
  average strategy), NOT from the current regret-matching iteration. This was
  the bug that made `solve_1v1_step` write oscillating values.

Release profile (`cote_cfr/Cargo.toml`):
```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
```
The build adds `RUSTFLAGS="-C target-cpu=native"` (AVX-512/FMA on Ice Lake)
via the rebuild script; a venv marker forces a rebuild when the flag changes.

`cote_cfr/1v1_table.csv` — **the 1v1 table** (belief key
`hA,hB,to_move,own_bank,own_sh,R,turn`; turn≥7 → 7). Built on the server,
gitignored. The bot loads it at startup. `cote_cfr/1v1_table_old.csv` is the
previous build, kept locally (gitignored) for comparison.

## 8. Server scripts — `server/`

- `build_1v1_table.py` — monolithic 1v1 table build (phase 4 + ramp,
  γ=1.0, truncated horizon; checkpoints, resume, multiprocessing).
- `rebuild_1v1_table.sh` — one-shot, fully automated rebuild on a fresh
  server: system packages (apt), Rust (rustup), venv, engine build, then
  backup → smoke (with an automatic convergence gate) → full → bench → compare.
  Steps can be run individually (`--env | --backup | --smoke | --full |
  --bench | --compare`).
- `gen_value_data.py`, `train_value_net.py`, `value_leaf.py` — ReBeL: data
  generation, value-net training, network leaf value.

## 9. Harnesses and benchmarks

- `play.py` — headless "human/LLM vs Planner" interface: `new/move/view/end/session`,
  `--compact` for LLM. Honest observation (hidden info is never shown).
- `interactive.py` — terminal UI (legacy interactive flow).
- `cfr_vs_planner.py` — CFR bot vs Planner (3v3, 3v3 leaves).
- `test_1v1_duel_bot.py` — fast pure-1v1 duel benchmark, bot vs Planner
  (both seats, multiprocessing).
- `botvbot.py` — bot vs bot (agent v3 vs v2).
- `bot_selfplay.py` — Planner vs itself.

## 10. Tests — `tests/`

`test_cote_cfr.py` (engine), `test_layers.py` (belief model), `test_new_engine.py`
(rules), `test_interactive.py` (UI). Run: `python -m unittest discover -s tests`.

## 11. Models and checkpoints

- `table_out/` — trained value-nets (`vnet1/2/3.npz` + `.pt`).
- `runs/` — `play.py` sessions (gitignored).

## 12. opencode config — `.opencode/agent/`

Universal free subagents (`deepseek-v4-flash-free`, `big-pickle`,
`hy3-free`, `laguna-s-2.1-free`, `mimo-v2.5-free`, `nemotron-3-ultra-free`,
`nemotron-3.5-lightning-free`) — for autonomous subtasks (search, summary,
checks). Name = model on the `opencode` provider.

## 13. How it all fits together

```
rules.py ── rules reference (game, harnesses, belief)
   │
   ├── agent.py / agent_v2.py  ── Planner (benchmark), belief from infoset.py
   ├── cfr_bot.py ── CFR bot: re-solves via _cote_cfr, leaves = 1v1_table.csv
   ├── cote_cfr (Rust) ── MicroTree + solve_1v1_step → the 1v1 table
   ├── trunk.py ── 3v3 trunk for the hierarchical solver
   └── server/*.py ── table build, data generation, value-net training
```

Benchmark flow: the harness runs `rules.py`; each turn the bot re-solves a
subtree (`_cote_cfr`) with the belief (`infoset.py`) and table leaves; the
Planner uses heuristics. Win/loss is judged by `rules.py`.

## 14. Current state and open questions

- **Avg-profile value fix is in and verified.** `solve_1v1_step` now writes
  stable, converged values (verified: recomputed cells match the table within
  spread < 0.003 across 100–1600 iterations). Rebuilds are reproducible.
- **1v1 table rebuilt on the server** (converged at phase-4 iter 14,
  delta < 0.02), with the Rust tuning above. Old table kept as
  `cote_cfr/1v1_table_old.csv`.

### Open problem: the new table is not behaving correctly in the benchmark

CFR bot vs Planner, pure 1v1 duels (`test_1v1_duel_bot.py`, same seeds as the
old runs):

| table | aggregate CFR winrate | CFR-first | CFR-second | draws |
|---|---|---|---|---|
| old | 35% | 56% | 14% | 0% |
| new | 22% | 30% | 14% | **13%** |

Draws are new and suspicious: in RULES.md a draw only happens if both sides
refuse to attack (turtling), which signals suboptimal/buggy play. A traced
draw game shows **both bots playing `d=4` shields every turn, never attacking**
for 61 half-turns.

Root-cause investigation so far (all verified by direct micro-tree solves):

- The micro-tree value on turn 1 **oscillates with search depth** for a fresh
  (3,3) duel: depth2=+0.403, depth3=+0.084, depth4=+0.267, depth6=+0.220,
  while the table says turn1=+0.403 (matches only depth 2). The bot uses
  depth=3 and therefore sees 0.084 and decides to bank/shield instead of
  attacking.
- At phase-4 turn 7 the micro-tree converges to the table value (0.997 ≈
  0.990), so phase 4 is a genuine fixed point.
- The table's turn 1..6 are built as a **cascade** `turn k ← turn k+2`
  (one backward-induction pass each), unlike phase 4 which iterates to
  convergence. Hypothesis (not yet fully proven): this cascade is the defect —
  deep micro-tree values on turn 1 disagree with the table, and draws/weak
  play follow.

Next steps (for a new session):
1. Verify the ramp-cascade hypothesis with a clean experiment: for turn k,
   compare the table value against a **deep** micro-tree solve with a cap that
   keeps the leaves inside the table (the earlier check used cap=4 which
   truncated leaves — invalid; redo with a cap that does not clip).
2. If confirmed, fix `build_1v1_table.py` to iterate each ramp turn to
   convergence (like phase 4) instead of a single cascade pass; re-smoke,
   re-bench.
3. Re-run `server/rebuild_1v1_table.sh --all` on the server and compare
   benchmark winrate and draw rate vs the acceptance criterion (>50%, no
   suspicious draws).

### Other notes

- γ: the table is built with γ=1.0 (truncated horizon); the bot re-solves with
  γ=0.995 — a search parameter, not a rule.
- Lab solver removed: `solver_tree_fh.py`, `match_1v1.py`, `run_full_history.py`,
  `run_table.py`, `export_1v1_table.py`, `test_1v1_solver_vs_planner.py`,
  `server_out_fh/` are gone. The table CSV is the only 1v1 artifact.
