# COTE Megaverse

The project is a human-fair, hidden-intent turn-based combat engine.

## Source Layout

- `src/cote_megaverse/rules.py`: authoritative immutable rules and transitions.
- `src/cote_megaverse/agent.py`: public-history belief model and bounded planner.
- `src/cote_megaverse/cli.py`: position-analysis command line entry point.
- `tests/test_new_engine.py`: focused tests for the new engine.

## Information Model

Public: character types, HP, ATK, active/alive characters, round, action
budget, bonus bank and resolved results of previous exchanges.

Hidden until resolution: current opponent shields and current allocation.
The planner must use a shield belief, never an exact hidden shield count.

## Rules

- Teams contain three characters of types `A`, `B`, `C` or `D`.
- Type advantage is `A > B > C > D > A` with multipliers `1.3` and `0.7`.
- Actions are unordered allocations of attack, defend and bonus.
- Every action budget must be spent.
- Defend protects the opponent's next turn, not the current exchange.
- Bonus is public and capped at four.
- A voluntary switch costs one action and can happen once per round.
- Death causes free forced promotion and does not consume voluntary switch.

## Commands

```powershell
python -m unittest tests.test_new_engine -v
python -m src.cote_megaverse.cli --team A,B,C --opponent B,C,C --depth 3
python -m py_compile src/cote_megaverse/rules.py src/cote_megaverse/agent.py src/cote_megaverse/cli.py
```

Do not reintroduce genome evolution, perfect-information policies, or legacy
compatibility wrappers without an explicit design decision.
