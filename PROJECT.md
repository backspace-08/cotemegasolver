# COTE Megaverse — структура проекта

Этот документ объясняет устройство репозитория: что где лежит, зачем нужно,
как части связаны и как всё работает вместе. Правила игры — в `RULES.md`.

---

## 1. Цель проекта

Построить сильный ИИ для пошаговой игры COTE Megaverse (3 на 3, скрытые
щиты/банк). Составляющие:

- **1v1-солвер** — точные значения равновесия дуэлей (для листьев 2v2/3v3).
- **Бот (CFR)** — играет через Rust-движок, ресолвит поддерево с верой.
- **Planner** — сильный эвристический противник (бенчмарк).
- **Инфраструктура обучения (ReBeL)** — value-net для belief-условных листьев.

---

## 2. Правила игры

`RULES.md` — точное описание правил: ходы, бюджет, банк, щиты, свитч, урон,
что видно и что скрыто, раскрытие скрытого, окончание матча.

---

## 3. Игровое ядро — `src/cote_megaverse/rules.py`

**Авторитетная реализация правил.** Иммутабельный симулятор:
`GameState`/`Side`/`Character`, `apply()` (разрешение хода), `prepare()`
(слив банка в бюджет), `initial()` (случайная генерация команд),
`exchange_damage()` (урон, округление к 100), `multiplier()` (цикл типов),
`base_budget()` (расписание ходов), `legal_allocations()`.

Все симуляции (игра, боты, харнессы) идут через `rules.py`.

## 4. Модель информации — `observation.py`, `infoset.py`

- `observation.py` — публичное наблюдение: что одна сторона может знать.
  Скрытые поля (щиты/банк противника) не пересекают границу.
- `infoset.py` — `OpponentModel`: вера (распределение) над скрытым сплитом
  `(щиты, банк)` противника, выводимая только из публичных фактов:
  бюджет, атаки, свитч, остаток `R = бюджет − атаки − свитч`, раскрытия.

## 5. Противники

- `agent.py` — **Planner (v3)**, основной эвристический противник: belief-модель,
  тактические факты (килл-пороги, урон), depth-limited поиск (alpha-beta),
  стратегические слои (наказать банкинг, бурст-сетап, выживание). Бенчмарк.
- `agent_v2.py` — **Planner (v2)**, замороженный вариант; используется в
  `botvbot.py` для сравнения поколений.
- `cfr_bot.py` — **CFR-бот**: ресолвит поддерево через Rust-движок
  (`_cote_cfr.MicroTree` / `solve_micro_belief`), листья дуэлей — из таблицы
  1v1, 2v2/3v3 — материал (или value-net, если задан).

## 6. Солверы

- `trunk.py` — **3v3-ствол** для CFR-D / subgame-солвера: весь матч, свитчи,
  промоушены, банки, щиты; при сведении к 1v1 отдаёт значение в лист.
- `strategy.py` — подсчёт ценности свитча, маржинальной ценности банка,
  стратегические цели (`Objective`).

## 7. Rust-движок и таблица — `cote_cfr/`

`cote_cfr/src/lib.rs` (PyO3, `import _cote_cfr`):

- `MicroTree` / `solve_micro` / `solve_micro_belief` — depth-limited CFR,
  микро-деревья, belief-корни, листья.
- `solve_1v1_step` — батч-шаг backward induction для сборки таблицы 1v1
  (фрагментами, multiprocessing-драйвер `server/build_1v1_table.py`).
- `load_1v1_table` — загрузка belief-таблицы.

`cote_cfr/1v1_table.csv` — **готовая таблица 1v1** (belief-ключ
`hA,hB,to_move,own_bank,own_sh,R,turn`; turn≥7 → 7). Собрана на сервере,
gitignored. Бот грузит её при старте.

## 8. Серверные скрипты — `server/`

- `build_1v1_table.py` — сборка монолитной 1v1-таблицы (фаза 4 + рампа,
  γ=1.0, усечённый горизонт; чекпоинты, resume, multiprocessing).
- `gen_value_data.py`, `train_value_net.py`, `value_leaf.py` — ReBeL:
  генерация данных, обучение value-net, листовое значение сети.

## 9. Харнессы и бенчмарки

- `play.py` — headless-интерфейс «человек/LLM против Planner»: `new/move/view/end/session`,
  `--compact` для LLM. Честное наблюдение (скрытое не показывается).
- `interactive.py` — терминальный UI (старый интерактивный поток).
- `cfr_vs_planner.py` — CFR-бот против Planner (3v3, 3v3-листья).
- `test_1v1_duel_bot.py` — быстрый бенчмарк чистых 1v1-дуэлей бот против
  Planner (обе роли, multiprocessing).
- `botvbot.py` — бот против бота (agent v3 vs v2).
- `bot_selfplay.py` — Planner против себя.

## 10. Тесты — `tests/`

`test_cote_cfr.py` (движок), `test_layers.py` (модель веры), `test_new_engine.py`
(правила), `test_interactive.py` (UI). Запуск: `python -m unittest discover -s tests`.

## 11. Модели и чекпоинты

- `table_out/` — обученные value-net (`vnet1/2/3.npz` + `.pt`).
- `runs/` — сессии `play.py` (gitignored).

## 12. Конфиг opencode — `.opencode/agent/`

Универсальные бесплатные субагенты (`deepseek-v4-flash-free`, `big-pickle`,
`hy3-free`, `laguna-s-2.1-free`, `mimo-v2.5-free`, `nemotron-3-ultra-free`,
`nemotron-3.5-lightning-free`) — для автономных подзадач (поиск, резюме,
проверки). Название = model на провайдере `opencode`.

## 13. Как всё работает вместе

```
rules.py ── эталон правил (игра, харнессы, вера)
   │
   ├── agent.py / agent_v2.py  ── Planner (бенчмарк), belief из infoset.py
   ├── cfr_bot.py ── CFR-бот: ресолв через _cote_cfr, листья = 1v1_table.csv
   ├── cote_cfr (Rust) ── MicroTree + solve_1v1_step → таблица 1v1
   ├── trunk.py ── 3v3-ствол для иерархического солвера
   └── server/*.py ── сборка таблицы, генерация данных, обучение value-net
```

Поток бенчмарка: харнесс гоняет `rules.py`; на каждый ход бот ресолвит
поддерево (`_cote_cfr`) с верой (`infoset.py`) и листьями таблицы; Planner
считает эвристиками. Победа/поражение — по `rules.py`.

## 14. Известные допущения и открытые вопросы

- **Модель урона — унифицирована на `rules.py`.** Rust-движок
  (`Trunk::per_hit_units` + `Trunk::hits_to_kill`) и `trunk.py` считают урон
  как `rules.py`: **per-hit** —
  `round(atk·mult/100)·100` за удар (Python `round()`, banker's half-to-even),
  всего `per_hit·hits`. Урон линеен по hits, `hits_to_kill = ceil(hp/per_hit)`
  точен. Пересборка таблицы 1v1 не нужна: она считается при atk=2000/mult=1.0,
  где `round(2000/100)·100 = 2000` тождественно (без погрешности).
- γ: таблица построена с γ=1.0 (усечённый горизонт); бот ресолвит с γ=0.995 —
  параметр поиска, не правило.
