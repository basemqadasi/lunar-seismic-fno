#!/usr/bin/env bash
set -u

STRICT=0
RUN_INSTALL_CHECK=0

for arg in "$@"; do
  case "$arg" in
    --strict)
      STRICT=1
      ;;
    --with-install)
      RUN_INSTALL_CHECK=1
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: ./release_check.sh [--strict] [--with-install]

Options:
  --strict        Treat metadata placeholders as failures.
  --with-install  Run editable install check: pip install -e .
USAGE
      exit 0
      ;;
    *)
      echo "Unknown option: $arg"
      exit 2
      ;;
  esac
done

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

pass() {
  PASS_COUNT=$((PASS_COUNT + 1))
  echo "[PASS] $1"
}

fail() {
  FAIL_COUNT=$((FAIL_COUNT + 1))
  echo "[FAIL] $1"
}

warn() {
  WARN_COUNT=$((WARN_COUNT + 1))
  if [[ "$STRICT" -eq 1 ]]; then
    fail "$1 (strict mode)"
  else
    echo "[WARN] $1"
  fi
}

run_check() {
  local label="$1"
  local cmd="$2"
  if eval "$cmd" >/tmp/release_check_cmd.out 2>/tmp/release_check_cmd.err; then
    pass "$label"
  else
    fail "$label"
    sed -n '1,40p' /tmp/release_check_cmd.err
  fi
}

echo "== Release QA: lunar-seismic-fno =="

if [[ ! -d .git ]]; then
  echo "Run this script from repo root (missing .git)."
  exit 2
fi

# 1) Metadata files exist
for f in README.md CITATION.cff LICENSE pyproject.toml requirements.txt; do
  if [[ -f "$f" ]]; then
    pass "File exists: $f"
  else
    fail "Missing required file: $f"
  fi
done

# 2) Placeholder metadata checks
if rg -n "<your-org-or-user>|TODO|TBD|Author\"|Primary\"" CITATION.cff README.md >/tmp/release_check_placeholders.out 2>/dev/null; then
  warn "Placeholder metadata found in README/CITATION"
  sed -n '1,20p' /tmp/release_check_placeholders.out
else
  pass "No placeholder metadata patterns in README/CITATION"
fi

# 3) Repo state checks
if git status --porcelain | rg -q "^"; then
  warn "Working tree is not clean"
  git status --short
else
  pass "Working tree clean"
fi

run_check "Python syntax compile" "python -m py_compile \\$(find src -name '*.py' | tr '\n' ' ')"

# 4) CLI help checks (OpenMP-safe env)
run_check "Waveform CLI help" "MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 PYTHONPATH=src python -m lunar_fno.train.train_waveform --help"
run_check "Spectrogram CLI help" "MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 PYTHONPATH=src python -m lunar_fno.train.train_spectrogram --help"

# 5) Expected-results checks
for f in expected_results/waveform_seed_metrics.csv expected_results/spectrogram_seed_metrics.csv expected_results/early_stopping_summary.csv; do
  if [[ -f "$f" ]]; then
    pass "Expected results file exists: $f"
  else
    fail "Missing expected results file: $f"
  fi
done

run_check "Expected results seed set == {42,7,101,2025}" "python - <<'PY'
import pandas as pd
files = [
  'expected_results/waveform_seed_metrics.csv',
  'expected_results/spectrogram_seed_metrics.csv',
  'expected_results/early_stopping_summary.csv',
]
expected = {42,7,101,2025}
for f in files:
    df = pd.read_csv(f)
    seeds = set(int(x) for x in df['seed'].unique())
    if seeds != expected:
        raise SystemExit(f'{f}: seeds {sorted(seeds)} != {sorted(expected)}')
print('seed check ok')
PY"

# 6) Data leakage checks
if git ls-files | rg -n '\\.(npz|mseed|pth|pt|ckpt)$' >/tmp/release_check_tracked_large.out; then
  fail "Restricted/large artifacts tracked by git"
  sed -n '1,40p' /tmp/release_check_tracked_large.out
else
  pass "No restricted data/checkpoint extensions tracked"
fi

# 7) Optional editable install
if [[ "$RUN_INSTALL_CHECK" -eq 1 ]]; then
  run_check "Editable install (pip install -e .)" "python -m pip install -e ."
else
  echo "[INFO] Skipping install check (use --with-install to enable)."
fi


echo
echo "== Summary =="
echo "PASS: $PASS_COUNT"
echo "WARN: $WARN_COUNT"
echo "FAIL: $FAIL_COUNT"

if [[ "$FAIL_COUNT" -gt 0 ]]; then
  exit 1
fi

if [[ "$WARN_COUNT" -gt 0 ]]; then
  echo "Completed with warnings. Use --strict to fail on warnings."
fi

exit 0
