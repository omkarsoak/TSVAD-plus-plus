#!/usr/bin/env bash
# Evaluate a trained TS-VAD+ checkpoint on DIHARD3 eval.
# Writes $SAVE_PATH/res_rttm and prints DER at collars 0.25 and 0.00.
set -euo pipefail
cd "$(dirname "$0")"

DIHARD_ROOT=${DIHARD_ROOT:-/path/to/DIHARD3}
SAVE_PATH=${SAVE_PATH:-exps/dihard3_eval}
MODEL=${MODEL:?set MODEL=/path/to/model_XXXX.model}
MAX_SPEAKER=${MAX_SPEAKER:-8}
LEGACY_HEAD=${LEGACY_HEAD:-false}                       # true only for checkpoints trained before the head fix

DEV="$DIHARD_ROOT/third_dihard_challenge_dev/data"
EVAL="$DIHARD_ROOT/third_dihard_challenge_eval/data"

python main.py \
--train_list "$DEV/ts_dev.json" \
--eval_list "$EVAL/ts_eval.json" \
--train_path "$DEV" \
--eval_path "$EVAL" \
--ref_rttm "$EVAL/rttm/all.rttm" \
--save_path "$SAVE_PATH" \
--rs_len 4 \
--test_shift 0.5 \
--min_silence 0.32 \
--min_speech 0.00 \
--threshold 0.50 \
--n_cpu 16 \
--max_speaker "$MAX_SPEAKER" \
--legacy_head "$LEGACY_HEAD" \
--eval \
--init_model "$MODEL"
