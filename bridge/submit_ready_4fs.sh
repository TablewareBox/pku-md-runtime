#!/bin/bash
# Queue the production-ready boxes at 4 fs.
# Physical length matches the CSV: 0.2 ns NPT equilibration, then 1.0 ns production.
# At 4 fs that is 50000 + 250000 steps. A report every 250 steps is 1 ps.
# One equilibration is shared by every property of the same system, temperature and seed.
# Columns: system_id, pdb, temperature_K, seed, eq_ensemble, prod_ensemble,
#          pressure_bar, forcefield, property.
# A finished prod.log containing "status": "completed" is skipped.
# One worker per GPU. GPU 0 is left to the 2 fs convergence extension.
set -u
ROOT=/home/changjh/MD_projects/PKUGraduateThesis
# shellcheck source=resolve_exp_system.sh
source "$ROOT/bridge/resolve_exp_system.sh"
if [[ -z "${OUTROOT:-}" ]]; then
  if [[ -n "${RESULT_DIR:-}" ]]; then
    case "$RESULT_DIR" in
      /*) OUTROOT=$RESULT_DIR/ready_4fs_1p2ns ;;
      *) OUTROOT=$PWD/$RESULT_DIR/ready_4fs_1p2ns ;;
    esac
  else
    OUTROOT=$ROOT/runs/ready_4fs_1p2ns
  fi
fi
EQ_STEPS=50000
PROD_STEPS=250000
REPORT=250
EQ_CHECKPOINT=12500
PROD_CHECKPOINT=25000
TIMESTEP=4.0
GPU=$1
LIST=$2
mkdir -p "$OUTROOT"
current=""
eq_ok=0
while IFS=$'\t' read -r sid pdb temp seed eq_ensemble prod_ensemble pressure forcefield property; do
  [ -n "$sid" ] || continue
  key="$sid|T${temp}|seed${seed}"
  base="$OUTROOT/$sid/T${temp}/seed${seed}"
  if [ "$key" != "$current" ]; then
    current=$key
    eq_ok=0
    mkdir -p "$base"
    if grep -q '"status": "completed"' "$base/eq.log" 2>/dev/null; then
      echo "SKIP eq $key"
      eq_ok=1
    else
      echo "START eq $key gpu=$GPU $(date +%T)"
      if bash "$ROOT/bridge/run_merged_production.sh" --gpu "$GPU" \
          --pdb "$(resolve_exp_system "$pdb")" --forcefield "$forcefield" \
          --temperature-K "$temp" --seed "$seed" --ensemble "$eq_ensemble" \
          --pressure-bar "$pressure" --hardcore r12 --timestep-fs "$TIMESTEP" \
          --steps "$EQ_STEPS" --report-interval "$REPORT" --checkpoint-interval "$EQ_CHECKPOINT" \
          --barostat-interval 25 \
          --output-prefix "$base/eq" > "$base/eq.log" 2>&1 \
          && grep -q '"status": "completed"' "$base/eq.log"; then
        echo "DONE eq $key $(date +%T)"
        eq_ok=1
      else
        echo "FAIL eq $key $(date +%T)"
      fi
    fi
  fi
  if [ "$eq_ok" != 1 ]; then
    echo "SKIP prod $key $property because equilibration did not finish"
    continue
  fi
  out="$base/$property"
  mkdir -p "$out"
  if grep -q '"status": "completed"' "$out/prod.log" 2>/dev/null; then
    echo "SKIP prod $key $property"
    continue
  fi
  echo "START prod $key $property gpu=$GPU $(date +%T)"
  if bash "$ROOT/bridge/run_merged_production.sh" --gpu "$GPU" \
      --pdb "$(resolve_exp_system "$pdb")" --forcefield "$forcefield" \
      --temperature-K "$temp" --seed "$seed" --ensemble "$prod_ensemble" \
      --pressure-bar "$pressure" --hardcore r12 --timestep-fs "$TIMESTEP" \
      --steps "$PROD_STEPS" --no-relax --state-in "$base/eq.final.state.xml" \
      --report-interval "$REPORT" --checkpoint-interval "$PROD_CHECKPOINT" \
      --barostat-interval 25 --frames "$out/prod.frames.npz" \
      --output-prefix "$out/prod" > "$out/prod.log" 2>&1 \
      && grep -q '"status": "completed"' "$out/prod.log"; then
    echo "DONE prod $key $property $(date +%T)"
  else
    echo "FAIL prod $key $property $(date +%T)"
  fi
done < "$LIST"
echo "WORKER gpu=$GPU finished $(date +%T)"
