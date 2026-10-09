#!/bin/bash
# Queue packed experimental boxes through run_merged_production.sh.
# Each item is 0.2 ns equilibration then 1.0 ns production at 0.5 fs.
# Columns: system_id, pdb, temperature_K, seed, eq_ensemble, prod_ensemble.
# One worker per GPU. A finished prod.log containing "status": "completed" is skipped.
set -u
ROOT=/home/changjh/MD_projects/PKUGraduateThesis
# shellcheck source=resolve_exp_system.sh
source "$ROOT/bridge/resolve_exp_system.sh"
if [[ -z "${OUTROOT:-}" ]]; then
  if [[ -n "${RESULT_DIR:-}" ]]; then
    case "$RESULT_DIR" in
      /*) OUTROOT=$RESULT_DIR/ready_nvt_1p2ns ;;
      *) OUTROOT=$PWD/$RESULT_DIR/ready_nvt_1p2ns ;;
    esac
  else
    OUTROOT=$ROOT/runs/ready_nvt_1p2ns
  fi
fi
EQ_STEPS=400000
PROD_STEPS=2000000
REPORT=2000
GPU=$1
LIST=$2
mkdir -p "$OUTROOT"
while IFS=$'\t' read -r sid pdb temp seed eq_ensemble prod_ensemble; do
  [ -n "$sid" ] || continue
  eq_ensemble=${eq_ensemble:-nvt}
  prod_ensemble=${prod_ensemble:-$eq_ensemble}
  out="$OUTROOT/$sid/T${temp}/seed${seed}"
  mkdir -p "$out"
  if grep -q '"status": "completed"' "$out/prod.log" 2>/dev/null; then
    echo "SKIP $sid T=$temp seed=$seed"
    continue
  fi
  echo "START eq $sid T=$temp seed=$seed gpu=$GPU $(date +%T)"
  if ! bash "$ROOT/bridge/run_merged_production.sh" --gpu "$GPU" \
      --pdb "$(resolve_exp_system "$pdb")" --temperature-K "$temp" --seed "$seed" --ensemble "$eq_ensemble" \
      --steps "$EQ_STEPS" --report-interval "$REPORT" --checkpoint-interval 100000 \
      --output-prefix "$out/eq" > "$out/eq.log" 2>&1; then
    echo "FAIL eq launch $sid T=$temp seed=$seed"
    continue
  fi
  if ! grep -q '"status": "completed"' "$out/eq.log"; then
    echo "FAIL eq $sid T=$temp seed=$seed"
    continue
  fi
  echo "START prod $sid T=$temp seed=$seed gpu=$GPU $(date +%T)"
  if ! bash "$ROOT/bridge/run_merged_production.sh" --gpu "$GPU" \
      --pdb "$(resolve_exp_system "$pdb")" --temperature-K "$temp" --seed "$seed" --ensemble "$prod_ensemble" \
      --steps "$PROD_STEPS" --no-relax --state-in "$out/eq.final.state.xml" \
      --report-interval "$REPORT" --checkpoint-interval 200000 \
      --output-prefix "$out/prod" > "$out/prod.log" 2>&1; then
    echo "FAIL prod launch $sid T=$temp seed=$seed"
    continue
  fi
  if grep -q '"status": "completed"' "$out/prod.log"; then
    echo "DONE $sid T=$temp seed=$seed $(date +%T)"
  else
    echo "FAIL prod $sid T=$temp seed=$seed"
  fi
done < "$LIST"
echo "WORKER gpu=$GPU finished $(date +%T)"
