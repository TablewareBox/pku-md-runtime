#!/bin/bash
# Validate or submit the directory next to this script.
# Dry-run is the default and does not create a job.
#
#   IMAGE=<registry url> MACHINE_TYPE=<sku name> bash bohr/submit.sh
#   IMAGE=... MACHINE_TYPE=... SUBMIT=1 bash bohr/submit.sh
#
# Machine names come from: bohr batchjob machine list --choose-type gpu
# After the job succeeds:
#   bohr batchjob download <job_id> --dest ./result
# The destination must not already exist. results.tar is inside it.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${IMAGE:?set IMAGE to the image URL from the Bohrium image center}"
: "${MACHINE_TYPE:?set MACHINE_TYPE from: bohr batchjob machine list --choose-type gpu}"

args=(
  batchjob submit
  --name "${NAME:-pku-md}"
  --image "$IMAGE"
  --machine-type "$MACHINE_TYPE"
  --command "bash run.sh"
  --input "$HERE/input"
  --out-file results.tar
  --out-file results/job.log
  --max-run-time "${MAX_RUN_TIME:-24h}"
  --max-wait-time "${MAX_WAIT_TIME:-30m}"
)
if [[ -n "${PROJECT_ID:-}" ]]; then
  args+=(--project-id "$PROJECT_ID")
fi
if [[ "${SUBMIT:-}" == 1 ]]; then
  exec bohr "${args[@]}" --yes -o json
fi
exec bohr "${args[@]}" --dry-run -o json
