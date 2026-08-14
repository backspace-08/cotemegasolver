# Запуск сборки 1v1-таблицы на арендованном сервере

Цель прогона — собрать монолитную 1v1-таблицу (Option A): значения равновесия
всех 1v1-дуэлей в belief-ключе. Результат — CSV
`cote_cfr/1v1_table.csv` (~1.45M строк, ~60МБ), который затем загружает
`cfr_bot.py` / `gen_value_data.py` вместо старой кап-6 таблицы (в ней были
ложные ничьи и провалы в material-fallback на turn>=8).

Ожидаемое время: **~30-60 минут** (фаза 4 ≈ 20-30 мин, рампа ≈ 5 мин, CSV ≈ 1 мин),
с запасом влезает в один арендованный час.

## Что именно делает прогон

1. **Фаза 4 (turn >= 7, стационарный бюджет 4-8)** — backward induction с
   `gamma=1.0` и усечённым горизонтом: итерация `T_2` (depth-2 CFR с листьями =
   текущая таблица) = значения (2k)-плёсной игры; горизонт растёт на 2 хода за
   итерацию, итерации до `max-abs-delta < tol` (по умолчанию 0.02) или
   `--max-iters`. Сетка: hits 1..16 × mover × own_bank 0..4 × own_sh 0..8 × R 0..8
   = 207 360 belief-состояний.
2. **Рампа (turn 6..1)** — по одному точному backward-induction шагу на ход
   (листья = уже готовые значения turn+2).
3. **CSV** — belief-ключ `hA,hB,to_move,own_bank,own_sh,R,turn,value`;
   turn>=7 отображается на 7 (значения фазы 4 инвариантны к ходу).

Параллелизм — **multiprocessing** (не потоки!): потоки упираются в общий
аллокатор, процессы масштабируются линейно (проверено). Rust-функция
`solve_1v1_step` решает слайс сетки за вызов.

Чекпоинты в `server/ckpt_1v1/` после каждой итерации фазы 4 и каждого хода
рампы; при обрыве тот же запуск продолжает с последнего чекпоинта.

## Пошагово (Ubuntu 22.04 или 24.04, всё внутри tmux)

### 1. База системы
```bash
sudo apt update && sudo apt upgrade -y        # upgrade — обновить пакеты свежего VPS
sudo apt install -y git curl build-essential python3 python3-pip python3-venv python3-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
cargo --version        # проверить: должен печатать версию
```

### 2. Виртуальное окружение Python (важно!)
На Ubuntu 24.04 системный pip блокирует установку пакетов (PEP 668:
`externally-managed-environment`). Поэтому ставим maturin и движок в venv:
```bash
cd ~
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install maturin
```

### 3. Код
```bash
cd ~
git clone <URL вашего репозитория> megaverse
cd megaverse
# убедиться, что мы на ветке/коммите с новым драйвером:
git log --oneline -3
```

### 4. Сборка движка под сервер (Ice Lake → AVX-512)
```bash
cd cote_cfr
RUSTFLAGS="-C target-cpu=native" maturin build --release
python -m pip install target/wheels/*.whl
cd ..
python -c "import _cote_cfr; print('engine ok')"   # проверить импорт
```

### 5. Калибровка (необязательно, но дёшево — ~2-4 мин)
Малый грид (hits 1..8) на 2-3 итерации, чтобы измерить реальную скорость
итерации и число воркеров ДО большого прогона (layout малого грида записан в
чекпоинт, большой прогон его проигнорирует — путаницы нет):
```bash
tmux new -s calib
python server/build_1v1_table.py --workers 56 --hits-max 8 --max-iters 2 \
    --ckpt-dir server/ckpt_calib --out /tmp/calib.csv
# смотрим на "[phase4 iter ...] iter= NN.Ns": это время полной итерации малого
# грида (25% от полного). Полный грид будет ~4x. Если малый ~<60с — отлично.
```

### 6. Основной прогон
```bash
tmux new -s solver
python server/build_1v1_table.py --workers 56 --out cote_cfr/1v1_table.csv
```
- `--workers`: проверьте `nproc` (у Ice Lake-сервера обычно 64). 56 безопасно на 64GB.
- Первые 2-3 итерации смотреть: `delta` должна убывать, `iter= NN.Ns` — время
  итерации. Если итерация > ~3 мин — `Ctrl-C`, снизить `--workers` (например 32)
  и запустить снова (подхватит чекпоинты, ничего не потеряется).
- Обрыв SSH: перезайти, `tmux attach -t solver` — процесс жив.
- Если процесс всё же упал: запустить ту же команду — resume с последнего чекпоинта.
- Форс-перезапуск с нуля: `--force` (или `rm -rf server/ckpt_1v1`).

### 7. Выгрузка результата
```bash
# после "[done]":
head -3 cote_cfr/1v1_table.csv && du -h cote_cfr/1v1_table.csv
```
На ноуте:
```bash
scp <user>@<host>:~/megaverse/cote_cfr/1v1_table.csv cote_cfr/1v1_table.csv
```
Потом удалить сервер в панели управления. Таблица (gitignored, артефакт сборки)
подхватывается `cfr_bot.py`/`gen_value_data.py` автоматически.

## Параметры драйвера

`--workers` (0 = все CPU), `--max-iters` (35), `--tol` (0.02),
`--solve-iters` (CFR-итераций на дерево, 150), `--hits-max` (16, для калибровки
можно меньше), `--no-ramp`, `--no-dump`, `--force`, `--ckpt-dir`, `--out`.

## Чек-лист «не навернуть»

- Сборка строго `RUSTFLAGS="-C target-cpu=native" maturin build --release` — иначе
  без AVX-512 и медленнее.
- Запуск строго в `tmux`/`screen` — иначе обрыв SSH = потерянный час.
- Первые 2 итерации смотреть на delta/ETA: если итерация дольше ~3 мин —
  остановить, снизить `--workers`, продолжить с чекпоинта (не теряется прогресс).
- На 64GB RAM 56 воркеров безопасно (каждый ~50-200МБ).
