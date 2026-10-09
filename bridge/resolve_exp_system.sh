#!/bin/bash
# Map a queue pdb column or --pdb value onto EXP_SYSTEMS.
# Absolute paths are unchanged. A relative path is looked up under
# EXP_SYSTEMS, after stripping a leading artifacts/exp_systems/ prefix.
# Default EXP_SYSTEMS is the directory baked into the image.
resolve_exp_system() {
  local raw=$1
  if [[ "$raw" = /* ]]; then
    printf '%s\n' "$raw"
    return
  fi
  local base=${EXP_SYSTEMS:-/home/changjh/MD_projects/DeePDih/artifacts/exp_systems}
  if [[ -f "$base/$raw" ]]; then
    printf '%s\n' "$base/$raw"
    return
  fi
  local rel=$raw
  case "$rel" in
    artifacts/exp_systems/*) rel=${rel#artifacts/exp_systems/} ;;
  esac
  printf '%s\n' "$base/$rel"
}
