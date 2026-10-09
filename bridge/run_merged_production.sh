#!/bin/bash
# Stage the matched ADMP CUDA plugin, then run production_merged_sr.py.
# --gpu and --snapshot are consumed here. Every system and simulation setting
# is a flag of the Python script (or a key in --config). Defaults are the
# frozen double-precision NVT protocol. Add --ensemble npt for a barostat.
#
# Snapshot SHA256 of the default plugin:
# 6e43000c41a354928f210cd4a90ec08dcddf9340657268a5fb368d1587216491
#
#   bash run_merged_production.sh --gpu 0 --steps 10000
#   bash run_merged_production.sh --config prod.json --output-prefix "$GATE/run1"
set -euo pipefail
source /home/changjh/miniconda3/etc/profile.d/conda.sh
conda activate md

SNAP=/home/changjh/MD_projects/plugin_snapshots/dmff_match_20261008/cuda_mode3/libADMPPmePluginCUDA.so
ROOT=/home/changjh/MD_projects/PKUGraduateThesis/bridge
# shellcheck source=resolve_exp_system.sh
source "$ROOT/resolve_exp_system.sh"
ORIG=$PWD
GPU=0
PY_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --snapshot) SNAP="$2"; shift 2 ;;
    --help|-h)
      echo "shell options: --gpu N   --snapshot PATH_TO_libADMPPmePluginCUDA.so"
      echo "paths: EXP_SYSTEMS for --pdb; RESULT_DIR receives the default output"
      echo
      exec python "$ROOT/production_merged_sr.py" --help
      ;;
    *) PY_ARGS+=("$1"); shift ;;
  esac
done

# Python runs with cwd=bridge. Relative --pdb, --output-prefix, --state-in,
# --frames, --config, and --forcefield are anchored to the caller's directory
# first, so a Bohrium job can write results next to its unpacked input.
abs_from_caller() {
  local val=$1
  if [[ "$val" = /* ]]; then
    printf '%s\n' "$val"
  else
    printf '%s\n' "$ORIG/$val"
  fi
}
NEW=()
has_output=0
while [[ ${#PY_ARGS[@]} -gt 0 ]]; do
  arg=${PY_ARGS[0]}
  PY_ARGS=("${PY_ARGS[@]:1}")
  case "$arg" in
    --pdb|--output-prefix|--state-in|--frames|--config|--forcefield)
      val=${PY_ARGS[0]}
      PY_ARGS=("${PY_ARGS[@]:1}")
      if [[ "$arg" == "--pdb" ]]; then
        val=$(resolve_exp_system "$val")
      fi
      val=$(abs_from_caller "$val")
      if [[ "$arg" == "--output-prefix" ]]; then
        has_output=1
      fi
      NEW+=("$arg" "$val")
      ;;
    *)
      NEW+=("$arg")
      ;;
  esac
done
if [[ "$has_output" -eq 0 && -n "${RESULT_DIR:-}" ]]; then
  NEW+=(--output-prefix "$(abs_from_caller "$RESULT_DIR/merged_prod")")
fi

echo "$SNAP" >&2
sha256sum "$SNAP" >&2
STAGE=$(mktemp -d /tmp/merged_prod_plugins.XXXXXX)
cp -a "$CONDA_PREFIX/lib/plugins/." "$STAGE/"
cp -a "$SNAP" "$STAGE/libADMPPmePluginCUDA.so"
export OPENMM_PLUGIN_DIR="$STAGE"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"
exec python production_merged_sr.py --plugin-dir "$STAGE" "${NEW[@]}"
