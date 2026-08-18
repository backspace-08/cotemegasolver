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
- `cfr_bot.py` — **CFR bot**: Continual Subgame Resolving through the Rust
  engine (`_cote_cfr.MicroTree`). 1v1 leaves come from the phase-4 table;
  2v2/3v3 leaves are evaluated **natively in Rust** by `chain_leaf_value`
  (the ChainLeaf oracle) or by a Python value-net if provided. Root belief is
  the previous resolve's opponent reach, hard-filtered by public facts and
  blended with the soft observation memory.

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
  micro-trees, belief roots, leaves. `MicroTree` takes a `compress` flag
  (action-grid compression for 3v3, E≥6) and a `telemetry` flag.
- `chain_leaf_value` / `chain_rec` — **native Rust ChainLeaf** (3v3 leaf
  oracle): duel-chain DP through LIFO promotion, residual HP via the physical
  damage-exchange ratio, memoized per solve. One-time leaf pre-eval in `build`.
- `opponent_reach` — the opponent's average-strategy reach over their next
  split (the continual-resolving root belief).
- `solve_1v1_step` — batched backward-induction step for building the 1v1
  table (sliced, multiprocessing driver `server/build_1v1_table.py`).
- `load_1v1_table` — loads the phase-4 belief table.
- `MicroSolver::root_strategy` returns probabilities from the **average**
  profile and the value from `avg_profile_value()` (a backward pass under the
  average strategy), NOT from the current regret-matching iteration. This was
  the bug that made `solve_1v1_step` write oscillating values.
- `sigs_inplace` intent is served by a flat zero-alloc `sig_buf` +
  `info_offsets` (disjoint-field borrowing; no per-iteration heap allocs).
  Fixed a subtle bug where the uniform signature path (`0 * norm`) stayed 0,
  which silently zeroed reach/avg.

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

- `build_1v1_table.py` — **phase-4-only** table build (`V = T(V)`, γ=0.995,
  truncated horizon; checkpoints, resume, multiprocessing). Ramp turns 1..6
  are behind `--ramp` (reference only); the bot resolves them at runtime.
- `rebuild_1v1_table.sh` — one-shot, fully automated rebuild on a fresh
  server: system packages (apt), Rust (rustup), venv, engine build, then
  backup → full → bench → compare. Steps can be run individually
  (`--env | --backup | --full | --bench | --compare`). `--env` rebuilds the
  Rust wheel (delete `.venv/.cote_cfr_rustflags` to force).
- `gen_value_data.py`, `train_value_net.py`, `value_leaf.py` — ReBeL: data
  generation, value-net training, network leaf value.

## 9. Harnesses and benchmarks

- `play.py` — headless "human/LLM vs bot" interface: `new/move/view/end/session`,
  `--compact` for LLM, `--bot cfr|planner` (CFR solver default now integrated).
  Honest observation (hidden info is never shown).
- `interactive.py` — terminal UI (legacy interactive flow).
- `cfr_vs_planner.py` — CFR bot vs Planner (3v3, duplicate matching, reports
  winrate/ΔScore/half-turns/latency).
- `test_1v1_duel_bot.py` — pure-1v1 duel benchmark, duplicate matching, even
  profiles E/F/G/I, both seats, multiprocessing.
- `server/analyze_state.py` — construct an arbitrary board by hand and step
  the CFR bot, dumping belief/root-strategy/value per turn.
- `server/gold_2v2.py` — exact 2v2 minimax (ground truth for ChainLeaf).
- `server/validate_chain.py` — validate the Rust ChainLeaf vs 2v2 gold.
- `botvbot.py` — bot vs bot (agent v3 vs v2).
- `bot_selfplay.py` — Planner vs itself.

## 10. Tests — `tests/`

`test_cote_cfr.py` (engine), `test_layers.py` (belief model), `test_new_engine.py`
(rules), `test_interactive.py` (UI). Run: `python -m unittest discover -s tests`.

## 11. Models and checkpoints

- `table_out/` — trained value-nets (`vnet1/2/3.npz` + `.pt`).
- `runs/` — `play.py` sessions (gitignored).

## 12. How it all fits together

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

## 13. Current state and open questions

### Architecture: Continual Subgame Resolving (CSR)

The solver moved from a monolithic 7-turn table to **Continual Subgame
Resolving** (DeepStack/Burch reach style):

- **Offline:** only the **stationary phase-4 table** (turn ≥ 7, `V = T(V)`)
  is built — `cote_cfr/1v1_table.csv` (belief key
  `hA,hB,to_move,own_bank,own_sh,R`, turn clamped to 7). Built on the server
  via `server/build_1v1_table.py` (multiprocessing, checkpoints, resume),
  converged at phase-4 iter ~20, γ=0.995, solve_iters=100/150, tol=0.02.
- **Runtime:** turns 1..6 are re-solved on the fly at `depth = 7 − turn`;
  turn ≥ 7 re-solves at `depth = 3` (3v3) / `self.depth` (1v1), leaves = the
  phase-4 table. No intermediate turn grids are precomputed.

### 1v1 benchmark (`test_1v1_duel_bot.py`, duplicate matching, even profiles)

Fair near-even start positions found in the table (`|V| < 0.05`):

| Profile | Position | CFR winrate (vs Planner) | ΔScore |
|---|---|---|---|
| E | (13,16) @t7 | ~75% | +1.10 |
| F | (11,14) @t7 | ~64% | +0.56 |
| G | (12,15) @t7 | ~74% | +1.06 |
| I | (13,15) @t1 | ~51% | +0.04 |

Key findings:
- **Phase-4 (turn ≥ 7) is the solver's strong side** — 64–75% on even
  positions. The exact table dominates the heuristic Planner in the endgame.
- **Early game (turn 1–6) is a ~51% parity** — the opening is simple (small
  budgets), the shallow resolve gains little over the heuristic.
- **Symmetric 16v16 is a pure first-mover win** in the model (`fresh(16,16)
  = +0.92`); even positions must give the later mover a 2–3 hit handicap.
- Hidden-split starts (e.g. opponent holding R=3) are NOT fair as benchmarks:
  a concrete split is tempo-biased vs the table's split-average.

### 3v3 benchmark (`cfr_vs_planner.py`, duplicate matching)

| Config | CFR winrate | ΔScore | avg lat |
|---|---|---|---|
| depth-3, iters=80 (old leaf) | ~48% | −0.06 | 831ms |
| depth-3 + native ChainLeaf + physical exchange | **53.8%** (10k games) | +0.177 | 138ms |
| depth-4 | 50% (not better, 55× slower) | +0.20 | 5568ms |

- **ChainLeaf** (Bench-Aware Sequential Matchup Chain DP) is the 3v3 leaf
  oracle, **ported to Rust** (`lib.rs::chain_leaf_value`/`chain_rec`): chains
  1v1 duels through LIFO promotion, residual HP via a **physical damage
  exchange ratio** (`per_hit_loser/per_hit_winner`, replacing the old magic
  rho=0.7). Validated on the exact 2v2 minimax (MAE 0.019).
- **Performance work:** zero-alloc flat `sig_buf` (fixed a uniform-signature
  bug where `0*norm` stayed 0), one-time leaf pre-eval in `build`, action-grid
  compression. Worst-case depth-4: 7.0s → 3.6s; real-match avg latency
  **456ms → 73ms** (later 138ms at 10k scale).
- **depth-4 does not pay off** at iters=80: the bigger tree does not converge,
  so it plays no better than depth-3 and is far slower.

### Belief / hidden-info model (`infoset.py`)

- Max-entropy **uniform prior** over the `(shields, bank)` split (identical to
  the table builder), as the base of a **soft observation memory**.
- **Soft memory prior:** Laplace-smoothed evidence weights over `(shields,
  bank)` accumulated from PUBLIC facts only — a resolved attack (proves the
  shields they held) and a revealed budget (proves the bank they held, and the
  joint `(shields, bank)` when the previous split resolves uniquely). The
  prior TILTS toward the opponent's observed tendency (e.g. a turtler's
  repeated 4 shields → `(4,0)` mass grows) but **never hard-zeroes any split**,
  so no false certainty. Counts decay each observed turn (`decay=0.85`) so the
  belief tracks the CURRENT tendency, not ancient history.
- **Turn-local shield pin** (cleared every new opponent allocation, including
  switches): a resolved attack hard-constrains the CURRENT split only — legal,
  because the shields we met are the ones currently held. Cross-turn tendency
  lives in the soft memory, not in a hard pin.
- **Continual reach:** the opponent's average-strategy reach from the previous
  resolve becomes the next root belief (DeepStack), hard-filtered by the same
  public facts and **blended with the soft memory** (`reach × memory`,
  normalized) so observed tendency reaches the solve root even when the reach
  is present.

### Known limitations / open questions

1. **3v3 leaf blind spot — banked burst threat.** In 3v3, ChainLeaf evaluates
   the active duel's `pA` from the 1v1 table but does NOT model that an
   opponent holding a large team bank will burst `a8` next round. The 1v1 table
   handles this exactly; the 3v3 leaf does not. (Note: the old hard bank pin
   could force a false `(0,4)=1.0` certainty here; the soft memory caps that at
   ~0.485 and the bot no longer blindly attacks into a shield wall.)
2. **1v1/2v1 shield-deadlock draws (~11%).** When a game reduces to a 1v1/2v1
   endgame against a turtler holding 4 shields every turn, the depth-3 resolve
   sometimes deadlocks to the 80-half-turn cap instead of finding the
   multi-burst kill. The soft memory makes the bot LEARN the turtler (belief
   tilts to `(4,0)`), but in type-disadvantage matchups (e.g. bot at 0.7 mult
   needing two bursts, which need >3 plies) the shallow resolve and the
   best-responder opponent model still stall. Depth 4+ is not the answer
   (measured: depth-4 draws more, depth-5 is ~6s/hit latency, depth-6 does not
   terminate); more CFR iterations reduce draws slightly at 2x latency.
3. **Switch timing.** The bot switches, but sometimes too late (lets a fighter
   fall to ~1 HP before switching) — the fixed-LIFO chain leaf does not model
   voluntary pre-death switching value.
4. **Early game parity.** Turn 1–6 resolve is shallow (`7−turn`) and gains
   little over the heuristic; improving it (e.g. a cheap ramp for turns 1–3)
   is low-yield because the opening is simple.
5. **Symmetric first-mover dominance.** In a symmetric 3v3 the first mover
   has a large model advantage (second seat ~44% vs first ~64%).

### Verified internals

- `solve_1v1_step` writes stable converged values (avg-profile value, not the
  current iteration).
- 54 tests pass: `python -m unittest discover -s tests`.

### Other notes

- γ: table built and bot re-solves with γ=0.995 — a search parameter, not a
  rule.
- The table is **phase-4 only**; the old cascade ramp (turn 1..6 in the CSV)
  is dead weight and no longer read by the bot.
- Lab solver files (`solver_tree_fh.py`, `match_1v1.py`, `run_full_history.py`,
  `run_table.py`, `export_1v1_table.py`, `test_1v1_solver_vs_planner.py`,
  `server_out_fh/`) are removed. The table CSV is the only 1v1 artifact.
