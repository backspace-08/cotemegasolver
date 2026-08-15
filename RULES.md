# COTE Megaverse — Game Rules

Self-contained, exact rules of the game. Read this only and you can play,
understand what is visible vs hidden, and how everything gets revealed.

## 1. Overview

Two players, each with a team of **3 characters**. Players alternate turns.
Goal: kill **all 3** of the opponent's characters. First to do so wins.

- `turn` counts **half-turns** (each player's move): turn 1 = first player,
  turn 2 = second player, turn 3 = first player, etc.
- Who moves first is a match parameter (benchmarks alternate seats).

## 2. Characters

Each character has:
- **Type** `A`, `B`, `C`, or `D` (duplicates allowed).
- **ATK** (attack strength), **HP** (current health), **max_hp** (initial HP).

**Global stat ranges:** HP 3000–10000 (step 100), ATK 1000–3500 (step 100).
Any value in these ranges is valid in the game.

> The shipped code defaults to a narrower benchmark subset (HP 5700–6300,
> ATK 1900/2000/2100). Those are simply sampled values from within the global
> range, chosen to keep test suites small — they are not a game limit.

All stats of all characters are **public**: types, ATK, current HP, max_hp, and
their full history.

### Type multiplier

Strength cycle: **A → B → C → D → A** (A beats B, B beats C, C beats D, D beats A).

Attacker type X vs defender type Y:
- **×1.3** if X is stronger than Y (X is the next type after Y in the cycle);
- **×0.7** if X is weaker than Y (Y is the next type after X in the cycle);
- **×1.0** otherwise — identical types, or types two steps apart in the cycle
  (e.g. A vs C, B vs D).

Every distinct pair of types falls into exactly one of these three cases.

### Damage

Damage dealt per turn:

```
damage = round(ATK × multiplier × hits / 100) × 100
```

The total is rounded to the nearest 100 (Python `round()` behavior). Damage is
always a multiple of 100. 0 hits → 0 damage.

## 3. Action budget

Each turn a player gets an **action budget** that must be spent **fully**
(no leftover actions allowed). An allocation that does not spend the whole
budget is invalid and rejected.

**Base budget** by `turn`:

| turn | base budget |
|------|-------------|
| 1    | 1           |
| 2–4  | 2           |
| 5–6  | 3           |
| 7+   | 4           |

Bank is added to the base:

```
budget = base(turn) + bank
```

Budget never exceeds **8**. Bank is capped at **4**.

## 4. Actions

The budget is split among:

1. **Attack** (attacks) — damage to the opponent's **active** character (the one
   currently in front of you). You can never target a bench character further in
   the stack, and never a dead character — only the active one is attackable.
2. **Shields** (defends) — see "Shields".
3. **Bank** (bonuses) — see "Bank".
4. **Switch** — change your active character.

`attacks + shields + bank` must equal the budget (minus 1 if a switch was made).
Any of the three can be 0 — a turn may be spent entirely on shields and/or bank,
with no attacks.

### Switch

- Costs **1 action**.
- At most **one switch per turn** (cannot switch back in the same turn).
- Only to a **living** own character, and **not** to the already-active one.
- The switched-to character becomes first in the team's stack order; the rest
  keep their relative order.
- The switch takes effect **before** the turn's attacks: if you switch, your
  attacks that turn use the new active character's ATK and type.

Example: with budget 8, spend 1 action on a switch and 7 on anything else.

### Promotion on death

If the active character dies, the first living character in the stack order
becomes active **for free** (no action cost). The promoted character becomes
first in the stack order; the rest keep their relative order. The initial stack
order is the team's listing order, active first.

Promotion is **not** a switch: it does not cost an action and does not consume
the one-switch-per-turn allowance.

## 5. Shields

- Shields placed on turn N block attacks on the **opponent's next turn** (N+1).
- One-to-one: `blocked = min(attacks, shields)`.
- After turn N+1 the shields **burn** — whether or not they blocked anything,
  even if no attack was made.
- Only the latest placement is active: shields do **not** stack across turns
  (the previous set is always burned before the new one matters).

## 6. Bank

- Bank is a hidden reserve that increases your budget on your **next** own turn.
- On that turn the bank is fully spent: `budget = base + bank`.
- Bank cap 4; total budget cap 8.

## 7. Attack resolution

When a player attacks:
1. Shields block: `blocked = min(attacks, shields)`.
2. `hits = attacks − blocked`.
3. Damage to the opponent's active character:
   `round(ATK × multiplier × hits / 100) × 100`.
4. HP drops; if HP ≤ 0 the character dies.
5. If the active character died, the next living one is promoted for free.

All attacks of a turn resolve in **one batch** against the defender's active
character at that moment. If that character dies, the promoted character is
**not** attacked this turn — the resolution is finished and the turn ends.

Only the defender can take damage on a turn; the attacker's own characters are
untouched by their attack. Both sides can therefore never die simultaneously.

## 8. What is visible and what is hidden

### Always public

- Turn number.
- Types, ATK, current HP, max_hp of all characters on both sides — current
  values and full history.
- The opponent's budgets of **completed** turns, their attack counts, their switches.
- Outcomes of all resolutions (hits landed, shields spent against you).

### Hidden — only at the moment you act

When it is your turn you **do not know** the opponent's latest placement:

- how many **shields** they currently hold (they will block your attack this turn);
- how much **bank** they put (it will increase their next budget).

Only their **sum** is known:

```
R = opponent_budget − opponent_attacks − opponent_switch
```

### Reveal — always and exactly

- **Bank** placed on turn N becomes public on turn N+2 (when the opponent moves
  again and their budget is shown). Always revealed, even if it was 0:
  `bank = budget − base`.
- **Shields** become public after they fire/burn — on your turn after the
  resolution. Always revealed, exactly: even if you did not attack, the
  opponent's spent shields are shown.

Because R and the spent shields are both public, the bank is in fact inferable
one turn earlier than N+2: `bank = R − spent_shields`.

### Conclusion

Exactly one value is hidden at any decision point — the opponent's latest
(shields, bank) placement; only its sum is known. Everything else is fully
public.

## 9. End of match

- The match ends when a side has no living characters left — the other side wins.
- There is **no built-in draw** and **no turn limit** by the rules. The game is
  designed to end in a kill.
- **The game is decisive under optimal play** — two sides each trying to win end
  the match in a kill. The mechanical reason:
  - Shield count is capped by your budget. At the base budget (4) you can place
    at most **4 shields**; 8 shields require a full bank (4), which itself takes
    turns to build up.
  - 8 shields are **not sustainable**: spending your whole budget on shields
    resets your bank to 0, so your next turn has only 4 actions — at most 4
    shields. To shield 8 again you must re-bank, and while banking you are back
    to at most 4 shields.
  - So a defender cannot block everything every turn: a banked burst of more
    than 4 hits always lands at least `attacks − 4` hits against a non-banked
    defender, and a defender who is banking is vulnerable in the meantime.
- A draw is mechanically possible only if both players deliberately refuse to
  attack (spending every turn on shields and/or bank) — such a line never deals
  damage and never wins, so it is never optimal. A draw therefore signals
  suboptimal or buggy play and is suspicious.
- Computational environments (bots, tests) may impose a technical turn limit to
  bound runtime. If a draw appears under such a limit, it should be
  investigated.
