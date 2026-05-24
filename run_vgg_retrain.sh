#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash run_vgg_retrain.sh [all|ind|ood] [extra python args...]

Examples:
  bash run_vgg_retrain.sh
  bash run_vgg_retrain.sh ind --overwrite
  METHOD_NAMES="UNDIAL SatImp" bash run_vgg_retrain.sh ood
EOF
}

if [[ "$#" -gt 0 && ( "${1}" == "-h" || "${1}" == "--help" || "${1}" == "help" ) ]]; then
  usage
  exit 0
fi

MODE="all"
if [[ "$#" -gt 0 && "${1}" != -* ]]; then
  MODE="${1}"
  shift
fi

require_file() {
  if [[ ! -e "${1}" ]]; then
    echo "Required file is missing: ${1}" >&2
    exit 1
  fi
}

require_command() {
  if ! command -v "${1}" >/dev/null 2>&1; then
    echo "Required command is not available: ${1}" >&2
    echo "Activate the prepared environment or install dependencies from requirements-llamafactory.txt / requirements-verl.txt." >&2
    exit 127
  fi
}

preflight() {
  require_command python
  require_command llamafactory-cli
  require_file src/test/vgg_test_retrain_ind.py
  require_file src/test/vgg_test_retrain_ood.py
  require_file src/LlamaFactory/examples/train_full/vgg.yaml
  require_file src/LlamaFactory/examples/deepspeed/ds_z3_config.json
}

run_ind() {
  bash scripts/vgg/vgg_test_retrain_ind.sh "$@"
}

run_ood() {
  bash scripts/vgg/vgg_test_retrain_ood.sh "$@"
}

case "${MODE}" in
  ind)
    preflight
    run_ind "$@"
    ;;
  ood)
    preflight
    run_ood "$@"
    ;;
  all)
    preflight
    run_ind "$@"
    run_ood "$@"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Expected one of: all, ind, ood" >&2
    exit 2
    ;;
esac
