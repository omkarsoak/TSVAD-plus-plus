#!/usr/bin/env bash
# Tiny end-to-end run of the real pipeline (train -> best/averaged checkpoint -> tune on val -> eval)
# for a laptop / CPU. Expects a mini DIHARD3 tree prepared with prepare_dihard.sh (same MAX_SPEAKER).
set -euo pipefail
cd "$(dirname "$0")"

export DIHARD_ROOT=${DIHARD_ROOT:?set DIHARD_ROOT=/path/to/DIHARD3_mini}
export MUSAN=${MUSAN:-/nonexistent} RIRS=${RIRS:-/nonexistent}   # optional; missing augmentation is skipped
export MAX_SPEAKER=${MAX_SPEAKER:-4}
export SAVE_PATH=${SAVE_PATH:-exps/smoke}
export BATCH_SIZE=${BATCH_SIZE:-2} MAX_EPOCH=${MAX_EPOCH:-3} WARM_UP_EPOCH=${WARM_UP_EPOCH:-3} N_CPU=${N_CPU:-2}

./run_train.sh
TRAIN_DIR="$SAVE_PATH" AVG_N=2 TEST_SHIFT=${TEST_SHIFT:-1} ./run_final.sh
