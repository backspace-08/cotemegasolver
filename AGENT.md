# COTE Megaverse

COTE Megaverse is a human-fair, hidden-intent, turn-based combat game, and the
subject of this project: **building an AI that beats a human at the highest
possible win rate**. This file is the normative specification. Code, UI,
planner, benchmark, replay, and tests must follow it. Reading it should give a
complete picture of the project: what the game is, how it plays, what is shown
when, and how the AI is built.

`GameState` is authoritative resolver state, not a public observation. It
contains secrets needed to resolve combat. A field being present in
`GameState` never implies that the opponent may see or use it.

---

## 1. What This Project Is

- A two-sided battle game. One side is the human, the other is the AI. Each
  side fields three characters with types and stats.
- The AI must win against humans with a high win rate. Humans win by making
  few errors and by reading the AI; the AI wins by playing soundly (punishing
  errors) while not being trivially readable.
- The game is **imperfect-information**: when a side acts, the opponent's
  shields and stored bonus are hidden. The heart of the game is committing
  attacks and shields blind against that hidden state.
- The AI is a **belief-based planner**, not a learned network: it maintains a
  distribution over the opponent's hidden state, applies hard tactical gates,
  scores candidate moves with strategic terms, and (only at genuine decision
  points) samples among near-equal moves. See §10.
- Tooling around the game lets anyone play against the AI (`play.py`), run
  sessions, track win rates, and run other AI models as test subjects
  (`play.py --run <name>` plus subagents). See §12.

### 1.1 Quick gameplay narrative

One match plays out like this:

1. Teams and stats are sampled (3 characters per side; HP 5700-6300, ATK
   1900-2100, types A/B/C/D). The first mover is a match parameter; in the
   interactive and CLI harness it is random per game.
2. On each turn the acting side gets an action budget and must spend every
   action. Actions are: attacks (deal damage), defends (shields that block the
   opponent's *next* turn), bonuses (bank actions for a later burst), and at
   most one switch (change the active character, costs 1).
3. Attacks resolve against the shields the opponent held from *its previous
   turn* — you commit blind. After resolution the outcome is public.
4. When an active character dies, the next living character is promoted for
   free; the dead one leaves the order entirely.
5. Kill all three of the opponent's characters to win. A match that hits the
   configured turn limit is a draw.

---

## 2. Match Model

- A match has two sides: human/player and AI/opponent.
- Each side has exactly three characters. Duplicate types are allowed.
- Each character has one type: `A`, `B`, `C`, or `D`.
- Initial HP is sampled independently from
  `5700, 5800, 5900, 6000, 6100, 6200, 6300`.
- Initial ATK is sampled independently from `1900, 2000, 2100`.
- `max_hp` is the character's sampled initial HP.
- A character is alive while `hp > 0`.
- A side loses when all three of its characters are dead.
- `turn` counts individual side turns, not pairs of turns.
- Either side may act first. Interactive play defaults to the human acting
  first, but the starting side is a match parameter, not a fixed rule. The
  `play.py` harness randomizes it per game.
- Moving first is a measured structural advantage, not a cosmetic detail. With
  identical policies on both seats, the first mover wins about 68 percent of
  games with a random policy and about 58 percent with a greedy policy.
  Therefore every benchmark must play each seed twice, once per starting side,
  and report the seats separately as well as combined. A win rate measured on a
  single seat is not a valid strength measurement.
- There is no built-in draw rule; matches end on a kill or at a configured
  turn limit (the benchmark uses `max_half_turns`, default 100).
- Pass a seeded `random.Random` to `initial()` when reproducible teams and
  stats are required.

---

## 3. Character Types

Type advantage forms this cycle:

```text
A > B > C > D > A
```

- Attacking the next type in the cycle uses multiplier `1.3`.
- Attacking the previous type in the cycle uses multiplier `0.7`.
- Every other matchup, including equal types, uses multiplier `1.0`.
- Type, ATK, current HP, max HP, alive/dead status, and active character are
  public information.

---

## 4. Turns And Action Budgets

Base actions are determined by the global turn number:

| Turn | Base actions |
| ---: | ---: |
| 1 | 1 |
| 2-4 | 2 |
| 5-6 | 3 |
| 7+ | 4 |

- At the start of the acting side's turn, its available actions are
  `min(8, base actions + stored bonus)`.
- Turn preparation always drains the entire stored bonus into the budget.
  Because base actions never exceed `4` and the bank is capped at `4`, the sum
  never exceeds the action cap, so preparation always leaves the bank at zero.
  A partially spent bank is unreachable, and during its own allocation a side
  always has bank zero and full bank capacity `4`.
- The stored bonus is capped at `4` and is hidden while it is stored.
- The acting side's action budget is public. Because preparation drains the
  whole bank, `bank = budget - base actions` becomes public at the moment the
  bank is spent, which is the owner's next turn. Banking is a delayed reveal,
  not a permanent secret: an action banked on turn `N` is visible on turn
  `N + 2`.
- Available actions are capped at `8`.
- Every available action must be spent. Passing or leaving actions unused is
  illegal.
- Actions form one unordered `Allocation`; they are not resolved as a command
  sequence.

An action can be allocated to:

- `attack`: one attempted attack in the current exchange;
- `defend`: one shield held against the opponent's next turn;
- `bonus`: one stored action for a later turn, up to the bonus cap;
- `switch`: one voluntary active-character change, costing one action.

The allocation invariant is:

```text
attacks + defends + bonuses + (1 if switched else 0) == available actions
```

---

## 5. Switches, Order, And Death

- A voluntary switch may target any living, non-active character.
- A voluntary switch costs one action.
- At most one voluntary switch may be selected in an allocation.
- A switch happens before that allocation's attack damage is calculated.
- Therefore attacks in the same allocation use the newly active character's
  type and ATK.
- A voluntary switch moves the selected character to the front of
  `stack_order`; it remains at the front for the rest of the match.
- If an active character dies, the next living character is promoted for free.
- Forced promotion does not consume an action and is not a voluntary switch.
- The promoted character is the first living character in the **target's own**
  stack order (never the actor's order).
- The dead active character is **removed from the order entirely**. The rebuilt
  stack is `[promoted] + remaining living characters in their previous relative
  order`. No dead entries and no duplicates ever appear in `stack_order`.
- The UI always renders the active character first. Other living characters
  follow the real `stack_order`; display order is part of game state, not a
  cosmetic sort.
- If no living character remains, the match ends immediately.

---

## 6. Shields And Resolution

Shields have a one-opponent-turn lifecycle:

1. A side allocates `defends` on its turn.
2. Those `defends` become that side's held shields after its allocation
   resolves.
3. They do not protect against attacks in the allocation where they were
   created.
4. They protect against the opponent's next allocation.
5. Each shield blocks at most one attempted attack.
6. All held shields are cleared after that opponent allocation resolves,
   whether or not the opponent attacked and whether or not every shield
   blocked an attack.

For an attack exchange:

```text
blocked = min(attempted attacks, defender held shields)
landed = attempted attacks - blocked
```

- Shields cannot reduce landed attacks below zero.
- Defends selected by the attacker become future shields only after current
  damage resolution.
- Damage cannot reduce HP below zero.

---

## 7. Damage And Kill Arithmetic

Damage is computed for the entire exchange and rounded exactly once:

```text
raw damage = attacker ATK * type multiplier * landed attacks
damage = round(raw damage / 100) * 100
```

- Python's `round` behavior used by `rounded_damage()` is authoritative.
- Zero landed attacks deal zero damage.
- Never calculate exchange damage as `rounded single-hit damage * landed`.
- Resolver, planner facts, switch calculations, replay, and UI must use
  `exchange_damage()` rather than duplicate the formula.
- Displayed damage is capped at the target's HP for the completed resolution;
  resolver HP is clamped to zero.

Kill preview has a deliberately narrower meaning:

- It is the minimum number of landed, unblocked attacks whose aggregate damage
  reaches the target's current HP.
- It ignores held, possible, and hypothesized opponent shields.
- It must use the same aggregate rounding as the resolver.
- Example: `2100 ATK * 0.7` against `6000 HP` deals `5900` with four landed
  attacks and `7400` with five. The preview is `5 attacks needed`.
- Internal planner calculations may add hypothesized blocked attacks when
  evaluating a shield belief. Such totals must never replace the UI's landed
  attack preview.

---

## 8. Information Model

### 8.1 Public Before An Allocation

Both sides may know:

- all character types, ATK, current HP, max HP, and alive/dead status;
- both active characters and stack orders;
- turn number and acting side;
- the acting side's public action budget, and therefore the bank it just
  drained;
- all previously resolved combat results and visible switches.

Each side also knows its own held shields.

### 8.2 Hidden Before Resolution

A side must not know or receive the opponent's:

- currently held shield count;
- currently stored bonus bank;
- current allocation's attacks;
- current allocation's defends;
- current allocation's bonuses;
- current switch choice.

The allocation remains hidden while it is being chosen. Resolver access to
these values does not make them public.

Three channels leak information at three different delays, and the gap between
them is where bluffing lives:

- An attack reveals itself immediately and completely. Attacks are public on
  resolution, so a full-budget attack proves a remainder of zero.
- A bank is revealed when it is spent, not when it is taken. An action banked on
  turn `N` becomes public on turn `N + 2` through the budget.
- A shield is revealed on every resolution, whether it was hit or expired
  unused. The spent shield count is public each turn, including zero and turns
  with no attack: a wasted shield is visible, so an opponent who shielded into
  a banking player is seen to have wasted it.

### 8.3 Public After Resolution

- A visible switch and its target become public.
- If attacks were attempted, total attacks, shields encountered, landed hits,
  and resulting damage become public.
- If no attack occurred, the exact composition of the opponent's allocation
  (how the remainder split into defends vs bank) is not printed; public state
  changes may still reveal a bonus or a switch normally.
- The defender's held shield count is revealed on every resolution, attack or
  not. On each of the acting side's allocations the UI reports the opponent's
  expired shield count (`AI spent X shields` / `bot held X shields`),
  including `X == 0` and turns with no attack. Wasted shields are always
  visible; the split into the bank is not.

---

## 9. Human-AI Fairness Contract

- Human and AI must make decisions from equivalent information boundaries.
- AI knows its own state and held shields, just as the human knows theirs.
- `Planner.choose()` must mask exact opponent shields even if a caller passes a
  full resolver `GameState`.
- Planner continuation must preserve chosen shields in resolver state so they
  block the next simulated exchange. Masking may affect decision input, but
  must never erase shields before resolution.
- At the search horizon, planner must still detect an immediately available
  legal match-ending move. A forced win or loss on the next allocation must
  not be replaced by a nonterminal HP evaluation merely because depth reached
  zero.
- Before resolution, AI uses legal-allocation hypotheses, public history, and a
  shield belief. It must not use exact hidden opponent shields.
- Public action accounting is mandatory, but it yields a remainder, not a
  shield count. Given the opponent's known budget, public attacks, and visible
  switch cost, `remainder = budget - attacks - switch cost` is the number of
  actions that went to defends or bonuses in unknown proportion. Deriving that
  remainder is fair inference, not resolver-state leakage.
- The remainder must never be treated as the held-shield count. Every split
  `defends + bonuses == remainder` is a distinct world. A remainder of `r`
  yields `r + 1` candidate shield counts, `0..r`. This ambiguity is the core of
  the game and must not be collapsed.
- A remainder of zero is the one exact case: the opponent holds no shields and
  banked nothing. Attacking with the full budget is therefore self-revealing.
  Spending every action on attacks buys damage at the price of total
  transparency on the following turn.
- Shield worlds are pruned by later public facts, not by assumption. A revealed
  budget of `b` on the owner's next turn proves the bank was `b - base actions`,
  which retroactively fixes the split of the earlier remainder. Every
  resolution reveals the defender's shield count, which also fixes the split.
  Until such a fact arrives, all splits stay live.
- Belief about currently held shields and prediction of the opponent's future
  allocation are separate models and must not be conflated.
- Because held shields are almost never exactly inferable at decision time,
  candidate moves must resolve in separate shield-world states. Continuation
  value combines probability-weighted expectation with downside risk; it must
  not resolve the move once against an invented zero-shield state.
- Shield-world priors must use opponent rationality, not a uniform split. A
  split that hands the mover a guaranteed lethal is evidence against itself
  when the opponent had a safe alternative.
- Safety gates are evaluated across all live shield worlds. A move is losing if
  some world permits a guaranteed loss, and lethal only if every world is
  lethal. Prefer a move that loses in no world over a move with a higher score
  that loses in one.
- Worlds are joint `(shields, bank)` hypotheses, not shield marginals: the bank
  sets the budget of the reply the planner must survive. The safety loss gate
  must size the opponent's reply from the *believed* bank, never from the
  resolver state with the bank masked to zero — masking would make every
  banked burst invisible to the gate, which is the bank-and-burst hole.
- The loss gate may special-case "only our active is our last living body": a
  side with a spare body cannot be wiped by a single allocation, so the reply
  test is a damage comparison, not a full reply enumeration.
- When no more than three characters remain alive across both sides, planner
  extends search beyond normal depth to reduce endgame horizon errors.
- Evaluation must account for material, living bodies, own bank, the believed
  distribution over the opponent's bank, next-turn action budgets,
  shield-adjusted pressure, immediate lethal, and preservation of a living
  character through switching.
- Tempo is a first-class term. An action banked is an action kept; an action
  attacked into a possible shield may be worth nothing. Evaluation must reward
  keeping future options, not only current HP difference.
- Expected incoming damage must use aggregate `exchange_damage()` for each
  hypothesized landed-attack count, never rounded single-hit damage multiplied
  by attacks.
- Root candidate selection uses hard tactical gates before heuristic ranking:
  guaranteed lethal, guaranteed immediate loss, and `kill + defense` are
  explicit tactical facts. Strategic score is only a tie-break among moves
  with comparable tactical safety.
- A guaranteed lethal plus defense outranks a bare guaranteed lethal when both
  finish the current target. A move that permits guaranteed immediate loss is
  excluded when a non-losing legal alternative exists.
- Bonus preparation and switch value may influence nonterminal choices, but
  cannot override an available tactical safety gate.
- When the opponent's remainder is exactly zero and no tactical safety gate
  applies, concrete damage pressure outranks passive bonus preparation. Do not
  concede free tempo by banking every action against a defenceless opponent.
- Planner and evaluation must never read the opponent's stored `bonus` field.
  Only a revealed budget may be used to reconstruct a past bank. Predicting the
  opponent's next budget requires a distribution over its hidden bank, not the
  resolver value.
- Public history may record attacks, revealed budgets, visible switches, and
  revealed shield counts.
- Structured `PublicEvent` history may be used for public facts, but it must
  never contain hidden allocation intent. Policy belief is a soft forecast of
  opponent style, not permission to use private state.
- Exact shields may enter planner history after any resolution, because they
  are revealed every resolution (attack or not).
- Interactive play, position analysis, benchmark, replay, and self-play must
  enforce the same boundary.
- Self-play must use independent planner histories and alternate sides through
  the same `apply()` transition used by interactive play.
- Never reintroduce perfect-information policies.

---

## 10. Planner Architecture

The planner (`agent.py` + `infoset.py` + `strategy.py`) is a deterministic
reasoning engine with one controlled source of randomness. It has four layers:

1. **Belief** (`infoset.py::OpponentModel`). The only sanctioned source of
   opponent hidden state. It keeps a joint distribution over `(shields, bank)`
   candidates paid out of the same public remainder, updated by revealed
   budgets, revealed shield counts, and attack exchanges. The prior over legal
   splits is **binomial**: each remainder action independently becomes a shield
   with probability `defend_share`, so evidence moves the belief sharply (an
   opponent repeatedly observed holding no shields is read as a banker). Attack
   observations feed that behavioural prior; an `EPSILON` keeps every legal
   world alive (splits are pruned only by hard public facts, never by
   assumption). It never reads the resolver's opponent fields.

2. **Gates** (hard, unconditional). Match-winning moves are taken first; moves
   that permit a guaranteed immediate loss in any joint `(shields, bank)` world
   are excluded when a safe alternative exists (`_reply_kills_us` sizes the
   opponent's reply from the believed bank, and `loss_probability` prefers a
   move that dies in no world over one that dies in a credible few);
   guaranteed-lethal-and-defense outranks a bare lethal. Dominated switches (to
   a strictly weaker body) are never sampled. These gates are checked before
   any scoring.

3. **Scoring** (heuristic, over the safe candidates). Material, living bodies,
   survival, tempo, switch value, burst setup, and four strategic terms:
   - `punish_banking`: if the opponent is exposed and banking, pressure with
     attacks instead of mirroring passivity;
   - `deny_burst`: if the opponent is banking a big burst, shields must be
     enough to survive its worst case — shields that cannot save are penalised
     (this breaks the "turtle", i.e. shielding 4 every turn forever);
   - `desperation`: when behind in living bodies, prefer threat over pure
     defense (no turtling to death);
   - `burst_setup`: banking is rewarded only when it converts the next turn
     into a lethal burst.

4. **Context-conditional mixing** (the only randomness). By default the bot is
   deterministic (argmax of the safe candidates). When `temperature > 0` and an
   RNG is set, it samples only among the moves that are genuinely close to the
   best (within `band_fraction` of the best score) and only when at least two
   such moves exist. This keeps the bot strong and hard to read without ever
   randomizing into a dominated move. `temperature == 0` is fully deterministic
   and is the default for tests.

`strategy.py::Objective` classifies the situation (`finish`, `survive`,
`deny_burst`, `break_stall`, `prepare_burst`, `normal`); it is advisory and is
only one term in the score. `switch_value` rewards switching to a body that
deals more damage to the opponent's active or that survives better than the
current active, and never recommends a strictly worse switch.

### 10.1 Self-play and reference tools

- `solver1v1.py` is the 1v1 GTO reference: hits-abstraction CFR on the
  reachable DAG with the stall=loss draw rule. It is not part of the live bot.
  Convergence is slow — the average strategy needs many iterations, and
  exploitability must be measured with `info_set_br_value` (the
  info-set-constrained best response); a naive per-state-max BR overstates and
  misreports it.
- `solver_tree_fh.py` is the same 1v1 game solved on the FULL tree (no state
  dedup) with perfect-recall info sets (each player's full observation
  sequence). Measured finding: the DAG solver's `info_key` collapses the hidden
  (shields, bank) split to a sum AND forgets the player's past observations,
  which makes the abstract game imperfect-recall — CFR then never converges
  (expl oscillates, ~0.12-0.6 even at 173k iters). With full-history info sets
  CFR converges: cap=6 hp=10000 reaches ~0.05 in ~300 iters and ~0.01 in
  ~500-600 (~20-25 min on 1 core; the hot passes are memory-bound, threads do
  not help a single run). Cost: the tree is much larger than the DAG (cap=6:
  ~5.1M nodes vs ~7.8k states); cap=7 (~100M) is beyond 64 GB.
  `server/run_full_history.py` trains it with checkpoints for resume.
  Both solvers accept `first_move` (0 = A acts first, 1 = B first; only A-first
  was supported originally) and explicit `start_states` (arbitrary mid-game
  positions with banks/shields set, for subgame starts in the 2v2/3v3
  hierarchy; `turn_cap` becomes max start turn + cap).
  `server/run_table.py` farms a grid of matchups × first_move across workers
  (each job is one independent solve; ~2-4 GB per cap=6 job) and can dump the
  per-state equilibrium value table (continuation values for the hierarchy).
- `match_1v1.py` runs solver-vs-bot matches (1v1, both seats, W/L/D).
- `botvbot.py` is a bot-vs-bot A/B driver (agent v3 vs the frozen v2 baseline);
  `bot_selfplay.py` plays the current bot against itself (both seats, depth 1)
  to measure first-mover advantage and draw rate at bot level.
- The converged cap=8 1v1 equilibrium (173k iterations, ~5.6 h) is archived in
  `artifacts/convergence_cap8/` (`checkpoint_final.pkl`, `final_results.txt`,
  `convergence_log.txt`).
- `strategy.py` and `agent.py` must remain human-fair; never reintroduce
  resolver-state reads for opponent secrets.
- `strategy.py` and `agent.py` must remain human-fair; never reintroduce
  resolver-state reads for opponent secrets.

---

## 11. Terminal UI Contract

### 11.1 State Preview

- Render human side, AI side, current turn, acting side, characters, HP bars,
  HP, ATK, types, active marker, and living stack order.
- Human-held shields may appear on the human active character.
- AI-held shields must never appear in normal state preview.
- Before a human allocation, show the human bonus meter and
  `Actions: base + bonus`. This is the human's own bank, which the human owns
  and may see.
- The AI's stored bank must never be rendered, in any turn, in any form. Only
  the AI's action budget on its own turn is public, and that budget reveals the
  bank it just drained. Never render an AI bonus meter, AI held shields, or an
  AI allocation.
- The acting side's budget is shown as `Actions: base + bonus` (base actions
  from the turn plus the bank revealed on its own turn).
- Matchup multiplier and potential one-attack aggregate damage may be shown.
- Show `[KILL possible] N attacks needed` only when the unblocked kill-preview
  count is within the human's currently available actions.
- Never include possible or current AI shields in kill-preview count.

### 11.2 Allocation Input

- Human commands are `a=atk`, `s=shield`, `b=bonus`, and `sw=switch`.
- Prompt until every available action has been allocated.
- Reject bonus allocation above the cap.
- Reject dead, active, nonexistent, duplicate, or otherwise illegal switch
  targets.
- The human naturally sees the allocation they are entering.
- Never print `AI selected: aN/dN/bN`, `move.label`, planner alternatives, or
  any equivalent AI allocation summary in interactive play.

### 11.3 Resolution Output

- Start a separate resolution block labeled `You turn #N` or `AI turn #N`.
- Show a resolved switch when one occurred.
- On every `You turn #N`, print the AI's held shield count (the shields held
  before this resolution), regardless of whether the human attacked. Never use
  the AI's newly selected future `defends`.
- The AI side observes the human's held shields on every resolution,
  symmetrically.
- Never print `You spent X shields`.
- A fully blocked attack prints `BLOCK!` and
  `<side>: N attacks vs X shields`.
- A partially blocked or unblocked attack prints total attempted attacks,
  resolved shields, landed hits, and aggregate damage.
- If no attack was allocated, print `<side> did not attack`, still reporting
  the opponent's held shield count.
- The completed exchange may reveal shields; this is a public result, not an
  information leak.

### 11.4 Match End

- The final move uses the same normal resolution block as every other move.
- After that block, do not render another state table or a `BATTLE OVER`
  preview.
- Do not show actions, matchup, potential damage, or kill preview after game
  over.
- Print only `YOU WIN`, `YOU LOSE`, or `DRAW` as the final result.

---

## 12. Harness And Tooling

### 12.1 `play.py` — the game harness

`play.py` is the headless harness used by a human, by this assistant, and by
test-subject subagents. It enforces the same fairness boundary as
`interactive.py` (no opponent shields or bank in the preview; shields revealed
on every resolution; budget shown as `Actions: base + bonus`).

Commands (all take `--run <name>` for per-session isolation):

- `new --seed N [--temp T] [--ai_first|--human_first]` — start a game. The
  first mover is random by default.
- `move --run <name> "a,d,b"[,sw]` — play your turn; the bot responds
  automatically. `a/d/b` need not sum to your budget: **the leftover is
  auto-banked as bonuses** (specify intent, the rest banks). Only going over
  budget errors. `sw` is a 1-based switch target among living characters.
- `move --run <name> -` — advance the bot's turn when it moves first.
- `view` / `end` — show the state / finish and record the result.
- `session --games N` — play N games back-to-back, each recorded
  automatically with a timestamp.
- `stats --run <name>` — summarize the run's W/L/D and per-seat split (see
  §12.2).
- The state display includes a `-- history (public) --` section with the
  last turns' public outcomes (attacks, damage, revealed shields), so a player
  can read the opponent's patterns without remembering every line.

Per-run isolation: state and logs live under `runs/<name>/` (`state.pkl`,
`session_log.json`, `winrate_log.json`). Parallel sessions never collide.

### 12.2 Win-rate statistics

`play.py stats --run <name>` reads `runs/<name>/winrate_log.json` and reports
total W/L/D plus a per-seat split (ai_first vs human_first). `session` mode
records a result automatically after each game. Sample guidance: ~100 games →
±10%, ~200 → ±7%.

### 12.3 Subagent test subjects

`.opencode/agent/` defines subagents that play the game as the human via
`play.py --run <name>`, so the AI can be evaluated against other AI models
(flagship and free models). Each subagent writes its results to its own
`runs/<name>/` folder as it goes, so a broken or interrupted subagent never
loses completed games. Because the CLI hides the opponent's shields and bank,
these tests measure real strength against the same information a human sees.

### 12.4 Source layout

- `src/cote_megaverse/rules.py`: authoritative immutable state, legal moves,
  damage, and transitions.
- `src/cote_megaverse/infoset.py`: opponent belief model (public-information
  shield/bank distribution).
- `src/cote_megaverse/observation.py`: public observation boundary.
- `src/cote_megaverse/agent.py`: planner — gates, scoring, mixing (current bot).
- `src/cote_megaverse/agent_v2.py`: frozen previous-generation planner, used
  only as the `botvbot.py` A/B baseline.
- `src/cote_megaverse/strategy.py`: objectives and switch valuation.
- `src/cote_megaverse/interactive.py`: human-fair terminal game and display
  contract.
- `src/cote_megaverse/benchmark.py`: seeded fair self-play, analysis, replay
  (`benchmark_policies`, `run_match`, `run_self_play`).
- `src/cote_megaverse/solver1v1.py`: 1v1 equilibrium solver — CFR on the
  reachable DAG with an HP→hits-to-kill abstraction and the stall=loss draw
  rule (see §10.1). `solve1v1.py` is the earlier laboratory solver, kept for
  reference only.
- `src/cote_megaverse/match_1v1.py`: solver-vs-bot match harness (1v1, W/L/D,
  both seats; maps abstracted hits to representative real HP for the bot).
- `src/cote_megaverse/cli.py`: position-analysis entry point.
- `play.py`: game harness (`new/move/view/end/session/stats`). `botvbot.py`:
  bot-vs-bot A/B driver. `bot_selfplay.py`: bot-vs-itself first-mover
  measurement.
- `tests/test_new_engine.py`, `tests/test_layers.py`,
  `tests/test_interactive.py`: rules, fairness, strategy, benchmark, and
  terminal-visibility tests.
- `.opencode/agent/`: subagent test subjects.

---

## 13. Verification And Commands

Run the complete suite after changes to rules, state, planner, visibility, or
UI:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
python -m src.cote_megaverse.interactive
python -m src.cote_megaverse.cli --team A,B,C --opponent B,C,C --depth 3
python -m py_compile src/cote_megaverse/rules.py src/cote_megaverse/agent.py src/cote_megaverse/observation.py src/cote_megaverse/interactive.py src/cote_megaverse/benchmark.py src/cote_megaverse/cli.py src/cote_megaverse/infoset.py src/cote_megaverse/solver1v1.py
git diff --check
```

Run a seeded AI-vs-human-policy baseline with the benchmark policies
(`random`, `greedy`, `bonus_shield`; every seed once per seat) and the exploit
policies `reader` (predicts the bot's shields) and `burster` (bank-and-burst)
to verify a change closes the hole a strong human exploited:

```powershell
python run_baseline.py --seeds 100 --limit 100 --workers 16 --out after_bench.txt
```

Do NOT parallelize with multiprocessing through `python -c "..."`: on Windows
spawn needs a file-based `__main__`, so drive it through `run_baseline.py`.
Do not interpret self-play win rate as human win rate. Record wins, losses,
draws, missed guaranteed lethals, and guaranteed-loss moves separately.

Evaluate against human-like play with the harness:

```powershell
python play.py session --games 20 --temp 0.12
python play.py stats --run default
```

Measure bot-level first-mover balance (bot vs itself, both seats):

```powershell
python bot_selfplay.py 40 8
```

Do not reintroduce genome evolution, LSTM/NumPy/Numba legacy planners, legacy
compatibility wrappers, or perfect-information behavior without an explicit
design decision and corresponding specification update.

---

## 14. Known Issues And History

Items that were measured bugs and have been fixed; keep them so regressions
are caught:

- Winner label in `benchmark.run_match` was inverted (`state.opponent` is the
  AI, so `opponent.lost` meant the AI lost, yet it was labelled the winner).
  Historical win-rate numbers from before the fix are void.
- Forced promotion used the actor's stack order and left dead/duplicate
  entries in the order. Now it promotes the first living character from the
  target's own order and rebuilds a clean stack (see §5).
- The CLI and the interactive UI disagreed on switch-target indexing
  (0-based vs 1-based). The CLI now takes 1-based targets.
- The harness and bot revealed the opponent's shields only on attack. Shields
  are now revealed on every resolution, symmetrically for human and AI
  (see §8).
- The `PublicEvent` rate properties referenced a nonexistent field and were
  dead code; they have been removed.
- `PublicHistory.observe` derived the opponent's exact shield count from the
  hidden bonuses the harness passed (budget - attacks - bonuses), leaking the
  current shields to the decision one turn early. It now records only public
  facts (attacks, budget, switch, remainder) and learns shields only from the
  per-resolution reveal. `observe_resolved` likewise never records bonuses.

Items that remain open (not bugs, but known engineering debt):

- The heuristic scoring terms are hand-tuned and overlapping; they are the
  calibration target if a principled utility replaces them.
- The 1v1 equilibrium solver is `solver1v1.py` (hits-abstraction CFR); its
  average strategy needs many iterations to converge, and exploitability must
  use the info-set-constrained best response (`info_set_br_value`).
  `solve1v1.py` is kept as the original laboratory.

Measured facts (do not re-derive blindly):

- **First-mover advantage in the full game is NOT the 1v1 GTO 69/31.** The
  1v1 equilibrium value is +0.30 for the first mover (≈69% of decided games)
  and the converged opening is ~99% a1/d0/b0 — but that is the *endgame*
  tempo effect, not the full-game balance. Symmetric full-game measurements:
  random-vs-random favors the second mover (≈44/56), greedy-vs-greedy favors
  the first (≈59/41), and bot-vs-bot (depth 1) is ~95% draws. Net: the full
  game is roughly balanced (≈50/50) and buffing the second player on the basis
  of the 1v1 number would likely over-buff. Decide a definition of "balance"
  (casual vs strong play) before changing anything.
- **Bot-vs-bot self-play at depth 1 is dominated by draws** (~95%), which
  makes it a weak signal for first-mover balance; the wall/defensive tendency
  must be addressed before self-play is informative. One bot game ≈ 0.28 s per
  ply, ~28 s/game, ~4.6 min for 80 games on 8 workers.
- The converged cap=8 1v1 run (173k iterations, ~5.6 h, explo ≈0.068/0.053)
  is archived under `artifacts/convergence_cap8/`; treat further multi-hour
  solver runs as high-cost and avoid repeating them casually.
