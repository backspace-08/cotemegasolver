#!/usr/bin/env bash
# Rebuild the 1v1 table after the avg-profile value fix, step by step.
#
# Steps (each can be run separately; --all runs them in order):
#   env      - verify engine + set up venv
#   backup   - save the current table for comparison
#   smoke    - small build on hits 1..6 + consistency check
#   full     - full build hits 1..16 (resumes from server/ckpt_1v1 on re-run)
#   bench    - CFRBot vs Planner duels (smoke then full)
#   compare  - diff old vs new table
#
# Usage (fresh server, from repo root):
#   bash server/rebuild_1v1_table.sh --all
#   # or stage by stage, e.g. after a crash:
#   bash server/rebuild_1v1_table.sh --full
#
# Notes:
#   - No --force on --full: build_1v1_table.py resumes from its checkpoint dir.
#   - First run creates .venv and builds the Rust wheel (needs Rust toolchain).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# ------------------------------------------------------------------ config
WORKERS="${WORKERS:-56}"                 # server 64 vCPU; laptop ~8
FULL_ITERS="${FULL_ITERS:-35}"           # phase-4 backward-induction iters
FULL_TOL="${FULL_TOL:-0.02}"
SOLVE_ITERS="${SOLVE_ITERS:-100}"        # micro-tree CFR iters per cell
SMOKE_ITERS="${SMOKE_ITERS:-3}"
SMOKE_HITS_MAX="${SMOKE_HITS_MAX:-6}"
SMOKE_WORKERS="${SMOKE_WORKERS:-8}"

# Rust build tuning. target-cpu=native lets LLVM use the full ISA of the build
# machine (e.g. AVX-512 on Ice Lake) for the CFR numeric loops. Keep the flag
# in sync with [profile.release] in cote_cfr/Cargo.toml.
RUST_TARGET_CPU="${RUST_TARGET_CPU:-native}"
RUSTFLAGS_CFR="-C target-cpu=$RUST_TARGET_CPU"

TABLE=cote_cfr/1v1_table.csv
TABLE_OLD=cote_cfr/1v1_table_old.csv
CKPT=server/ckpt_1v1
CKPT_TEST=server/ckpt_test
TABLE_TEST=server/table_test.csv

log() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }

# ------------------------------------------------------------------ env
# Install every system dependency automatically: apt packages (gcc for the
# Rust linker, python3-venv/pip, curl), Rust via rustup, a venv with numpy +
# maturin, then build the Rust engine. Runs on a bare Ubuntu/Debian VM.
step_env() {
  log "env: verify toolchain"

  # ---- system packages (apt) ------------------------------------------------
  local need_apt=0
  command -v gcc >/dev/null 2>&1 || need_apt=1
  command -v curl >/dev/null 2>&1 || need_apt=1
  command -v make >/dev/null 2>&1 || need_apt=1
  python3 -m venv --help >/dev/null 2>&1 || need_apt=1
  python3 -m pip --version >/dev/null 2>&1 || need_apt=1
  if [ "$need_apt" = "1" ]; then
    log "env: installing system packages (gcc, make, curl, python3-venv, python3-pip)"
    if ! command -v sudo >/dev/null 2>&1; then
      echo "sudo missing; run this script as root or install sudo first" >&2
      exit 1
    fi
    sudo apt-get update -qq
    sudo apt-get install -y -qq build-essential curl python3-venv python3-pip
  fi

  # ---- Rust via rustup if cargo is missing (fresh server) --------------------
  if ! command -v cargo >/dev/null 2>&1; then
    log "env: installing Rust via rustup"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
  fi
  command -v cargo >/dev/null || { echo "cargo still missing after rustup install"; exit 1; }

  # ---- venv + python deps ----------------------------------------------------
  if [ ! -d .venv ]; then
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --quiet --upgrade pip numpy maturin

  # ---- build the Rust engine ---------------------------------------------------
  # Rebuild when the engine is missing OR the target-cpu flag changed (the
  # venv marker records which flags the installed wheel was built with).
  local built_flag=""
  if [ -f ".venv/.cote_cfr_rustflags" ]; then
    built_flag="$(cat .venv/.cote_cfr_rustflags)"
  fi
  if ! python -c "import _cote_cfr" 2>/dev/null || [ "$built_flag" != "$RUSTFLAGS_CFR" ]; then
    log "env: building Rust engine (RUSTFLAGS=$RUSTFLAGS_CFR)"
    RUSTFLAGS="$RUSTFLAGS_CFR" python -m maturin build --release -m cote_cfr/Cargo.toml
    pip install --quiet --no-deps --force-reinstall cote_cfr/target/wheels/*.whl
    printf '%s' "$RUSTFLAGS_CFR" > .venv/.cote_cfr_rustflags
  else
    log "env: Rust engine already built with $RUSTFLAGS_CFR"
  fi
  python -c "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'src'); import _cote_cfr; print('engine OK')"
}

# ------------------------------------------------------------------ backup
step_backup() {
  log "backup: keep current table for comparison"
  if [ -f "$TABLE" ]; then
    cp "$TABLE" "$TABLE_OLD"
    echo "saved -> $TABLE_OLD ($(du -h "$TABLE_OLD" | cut -f1))"
  else
    echo "no current table at $TABLE; nothing to back up"
  fi
}

# ------------------------------------------------------------------ smoke
step_smoke() {
  log "smoke: build hits 1..$SMOKE_HITS_MAX ($SMOKE_ITERS iters, $SMOKE_WORKERS workers)"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python server/build_1v1_table.py \
    --ckpt-dir "$CKPT_TEST" --out "$TABLE_TEST" \
    --workers "$SMOKE_WORKERS" \
    --hits-min 1 --hits-max "$SMOKE_HITS_MAX" \
    --bank-max 4 --sh-max 8 --r-max 8 \
    --gamma 1.0 --solve-iters "$SOLVE_ITERS" \
    --max-iters "$SMOKE_ITERS" --tol 0.05 --force

  log "smoke: consistency check (turn1 <- turn3 must converge)"
  python - "$TABLE_TEST" "$SMOKE_HITS_MAX" <<'EOF'
import csv, sys
sys.path.insert(0, "."); sys.path.insert(0, "src")
import _cote_cfr

table, hits_max = sys.argv[1], int(sys.argv[2])
nh = hits_max - 1 + 1
def gidx(hA, hB, mv, bank, sh, r):
    return (((((hA-1)*nh + (hB-1))*2 + mv) * 5 + bank) * 9 + sh) * 9 + r

leaf3 = [0.0] * (nh*nh*2*5*9*9)
with open(table) as f:
    for row in csv.DictReader(f):
        if int(row["turn"]) == 3:
            leaf3[gidx(int(row["hA"]), int(row["hB"]), int(row["to_move"]),
                      int(row["own_bank"]), int(row["own_sh"]), int(row["R"]))] = float(row["value"])

key = gidx(2, 2, 0, 0, 0, 0)
vals = []
for it in (100, 1600):
    out, _ = _cote_cfr.solve_1v1_step(leaf3, start=key, end=key+1, root_turn=1,
                                      hits_min=1, hits_max=hits_max, bank_max=4,
                                      sh_max=8, r_max=8, gamma=1.0, solve_iters=it)
    vals.append(out[0])
spread = abs(vals[0] - vals[1])
print("turn1<-(turn3) iters=100/1600: %s spread=%.4f" % (vals, spread))
if spread > 0.01:
    print("FAIL: value is not converging; do NOT run the full build")
    sys.exit(1)
print("OK: converged")
EOF
}

# ------------------------------------------------------------------ full
step_full() {
  log "full: build hits 1..16 ($FULL_ITERS iters, $WORKERS workers, tol=$FULL_TOL)"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python server/build_1v1_table.py \
    --ckpt-dir "$CKPT" --out "$TABLE" \
    --workers "$WORKERS" \
    --hits-min 1 --hits-max 16 \
    --bank-max 4 --sh-max 8 --r-max 8 \
    --gamma 1.0 --solve-iters "$SOLVE_ITERS" \
    --max-iters "$FULL_ITERS" --tol "$FULL_TOL"
  # no --force: re-running resumes from $CKPT
  echo "done -> $TABLE"
}

# ------------------------------------------------------------------ bench
step_bench() {
  log "bench: CFRBot vs Planner 1v1 duels"
  # shellcheck disable=SC1091
  source .venv/bin/activate

  log "bench: smoke (16 games)"
  python test_1v1_duel_bot.py --games 16 --workers 4 --cfr-depth 3 \
    --cfr-iters 100 --cfr-cap 6 --pl-depth 2 --pl-max-nodes 2000 \
    --seed-start 7000 --tag duel_new_smoke

  log "bench: full (100 games)"
  python test_1v1_duel_bot.py --games 100 --workers 6 --cfr-depth 3 \
    --cfr-iters 100 --cfr-cap 6 --pl-depth 2 --pl-max-nodes 2000 \
    --seed-start 7000 --tag duel_new_100

  echo
  echo "Acceptance: aggregate CFR winrate must be > 50%; the collapse on the"
  echo "second seat (was 14-25%) should shrink toward the Planner's ~46%."
}

# ------------------------------------------------------------------ compare
step_compare() {
  log "compare: old vs new table"
  python - "$TABLE_OLD" "$TABLE" <<'EOF'
import csv, sys

old_p, new_p = sys.argv[1], sys.argv[2]
def load(p):
    d = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            k = (int(row["hA"]), int(row["hB"]), int(row["to_move"]),
                 int(row["own_bank"]), int(row["own_sh"]), int(row["R"]), int(row["turn"])
            d[k] = float(row["value"])
    return d
old, new = load(old_p), load(new_p)
common = set(old) & set(new)
diffs = [abs(old[k] - new[k]) for k in common]
if not diffs:
    print("no common keys")
    sys.exit(0)
import statistics
n = len(diffs)
big = sum(1 for d in diffs if d > 0.05)
print("keys: %d common, mean|diff|=%.4f, max|diff|=%.4f, |diff|>0.05: %d (%.2f%%)"
      % (n, statistics.mean(diffs), max(diffs), big, 100.0 * big / n))
EOF
}

# ------------------------------------------------------------------ main
ALL_STEPS="env backup smoke full bench compare"
RUN="${1:-}"

case "${RUN}" in
  --all)      for s in $ALL_STEPS; do "step_$s"; done ;;
  --env)      step_env ;;
  --backup)   step_backup ;;
  --smoke)    step_smoke ;;
  --full)     step_full ;;
  --bench)    step_bench ;;
  --compare)  step_compare ;;
  ""|-h|--help)
    echo "usage: $0 --all | --env | --backup | --smoke | --full | --bench | --compare"
    echo
    echo "steps: $ALL_STEPS"
    ;;
  *) echo "unknown step: $1"; exit 1 ;;
esac
