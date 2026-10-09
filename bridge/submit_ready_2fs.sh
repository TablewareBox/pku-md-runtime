#!/bin/bash
# Queue production-ready boxes at 2 fs. One worker per GPU.
# Columns: system_id, pdb, temperature_K, seed, eq_ensemble, prod_ensemble,
#          pressure_bar, forcefield, property.
# A finished prod.log containing "status": "completed" is skipped.
#
# Density uses the protocol fixed from BF-165EC-55Li-55FSI. Three seeds of a
# 0.2 ns equilibration plus ~0.1 ns of production agreed to 0.004 g/mL, and the
# running mean moved 0.0006 g/mL over the last 30 ps. The first ~40 ps is the
# compression transient, so density is 50 ps NPT equilibration (25000 steps)
# then 100 ps NPT production (50000 steps), one seed. A report every 500 steps
# is 1 ps. These lengths match the density rows of
# data/production_ready_experiments.csv.
#
# Viscosity and conductivity keep 0.2 ns equilibration and 1.0 ns production.
set -u
ROOT=/home/changjh/MD_projects/PKUGraduateThesis
PDBROOT=/home/changjh/MD_projects/DeePDih
OUTROOT=${OUTROOT:-$ROOT/runs/ready_2fs_1p2ns}
REPORT=500
TIMESTEP=2.0
GPU=$1
LIST=$2
mkdir -p "$OUTROOT"
current=""
eq_ok=0
while IFS=$'\t' read -r sid pdb temp seed eq_ensemble prod_ensemble pressure forcefield property; do
  [ -n "$sid" ] || continue
  key="$sid|T${temp}|seed${seed}"
  base="$OUTROOT/$sid/T${temp}/seed${seed}"
  if [ "$property" = "density" ]; then
    EQ_STEPS=25000
    PROD_STEPS=50000
    EQ_CHECKPOINT=25000
    PROD_CHECKPOINT=25000
  else
    EQ_STEPS=100000
    PROD_STEPS=500000
    EQ_CHECKPOINT=25000
    PROD_CHECKPOINT=50000
  fi
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
          --pdb "$PDBROOT/$pdb" --forcefield "$forcefield" \
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
      --pdb "$PDBROOT/$pdb" --forcefield "$forcefield" \
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
