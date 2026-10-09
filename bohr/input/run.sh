#!/bin/bash
# Container command for: bohr batchjob submit --command "bash run.sh"
# The worker unpacks --input into the job working directory and runs this
# there. Keep every product under ./results so --out-file can name it.
set -euo pipefail
cd "$(dirname "$0")"
WORKDIR=$PWD

if [[ -f job.env ]]; then
  # shellcheck disable=SC1091
  source ./job.env
fi

export RESULT_DIR="$WORKDIR/results"
mkdir -p "$RESULT_DIR"
if [[ -z "${EXP_SYSTEMS:-}" ]]; then
  if [[ -d "$WORKDIR/exp_systems" ]]; then
    EXP_SYSTEMS=$WORKDIR/exp_systems
  else
    EXP_SYSTEMS=/home/changjh/MD_projects/DeePDih/artifacts/exp_systems
  fi
  export EXP_SYSTEMS
fi

MODE=${MODE:-smoke}
GPU=${GPU:-0}
STEPS=${STEPS:-200}
LIST=${LIST:-list.tsv}
BRIDGE=/home/changjh/MD_projects/PKUGraduateThesis/bridge
LOG=$RESULT_DIR/job.log

{
  echo "mode=$MODE gpu=$GPU"
  echo "exp_systems=$EXP_SYSTEMS"
  echo "result_dir=$RESULT_DIR"
  date -u +%Y-%m-%dT%H:%M:%SZ
} | tee "$LOG"

finish() {
  local status=$?
  tar -C "$WORKDIR" -cf "$WORKDIR/results.tar" results || true
  exit "$status"
}
trap finish EXIT

case "$MODE" in
  smoke)
    args=(--gpu "$GPU" --steps "$STEPS" --output-prefix "$RESULT_DIR/smoke")
    if [[ -n "${PDB:-}" ]]; then
      args+=(--pdb "$PDB")
    fi
    bash "$BRIDGE/run_merged_production.sh" "${args[@]}" >>"$LOG" 2>&1
    ;;
  nvt|2fs|4fs)
    bash "$BRIDGE/submit_ready_${MODE}.sh" "$GPU" "$LIST" >>"$LOG" 2>&1
    ;;
  leg)
    ff=${FORCEFIELD:-/home/changjh/MD_projects/PhyNEO-Electrolyte/examples/md_simulation/phyneo_hmtff_admp.xml}
    common=(--gpu "$GPU" --pdb "$PDB" --forcefield "$ff"
      --temperature-K "$TEMP" --seed "$SEED"
      --pressure-bar "${PRESSURE:-1}" --hardcore "${HARDCORE:-r14}"
      --timestep-fs "$TIMESTEP")
    eq_extra=()
    if [[ "$EQ_ENSEMBLE" == npt ]]; then
      eq_extra=(--barostat-interval 25)
    fi
    bash "$BRIDGE/run_merged_production.sh" "${common[@]}" \
      --ensemble "$EQ_ENSEMBLE" "${eq_extra[@]}" \
      --steps "$EQ_STEPS" --report-interval "$REPORT" \
      --checkpoint-interval "${EQ_CHECKPOINT:-$EQ_STEPS}" \
      --output-prefix "$RESULT_DIR/eq" >>"$LOG" 2>&1
    prod_extra=()
    if [[ "$PROD_ENSEMBLE" == npt ]]; then
      prod_extra=(--barostat-interval 25)
    fi
    frames=()
    if [[ "${PROPERTY:-}" == conductivity ]]; then
      frames=(--frames "$RESULT_DIR/prod.frames.npz")
    fi
    bash "$BRIDGE/run_merged_production.sh" "${common[@]}" \
      --ensemble "$PROD_ENSEMBLE" "${prod_extra[@]}" "${frames[@]}" \
      --steps "$PROD_STEPS" --no-relax \
      --state-in "$RESULT_DIR/eq.final.state.xml" \
      --report-interval "$REPORT" \
      --checkpoint-interval "${PROD_CHECKPOINT:-$PROD_STEPS}" \
      --output-prefix "$RESULT_DIR/prod" >>"$LOG" 2>&1
    ;;
  *)
    echo "MODE must be smoke, nvt, 2fs, or 4fs" | tee -a "$LOG"
    exit 2
    ;;
esac
