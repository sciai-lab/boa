#!/bin/zsh
# wrapper_sweep.sh
# Usage: ./wrapper_sweep.sh <python_or_hydra_command> [args...]


LOG_ROOT="${BOA_MODELS}/train/slurm_multiruns"
COUNTER_FILE="${LOG_ROOT}/.multi_run_counter"
mkdir -p "$LOG_ROOT"

# Atomically read, increment, and store the counter
if [[ -f "$COUNTER_FILE" ]]; then
	COUNTER=$(<"$COUNTER_FILE")
else
	COUNTER=0
fi
COUNTER=$((COUNTER + 1))
echo "$COUNTER" >| "$COUNTER_FILE"

SWEEP_DIR="${LOG_ROOT}/run_${COUNTER}"
mkdir -p "$SWEEP_DIR"

# Pass sweep_dir as a hydra override, then run the rest of the command line
exec "$@" hydra.sweep.dir="$SWEEP_DIR"
