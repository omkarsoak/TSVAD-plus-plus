#!/usr/bin/env bash
# Train TS-VAD+ on DIHARD3 dev, evaluate on DIHARD3 eval.
# Edit the paths below (or export them before calling this script).
set -euo pipefail
cd "$(dirname "$0")"

DIHARD_ROOT=${DIHARD_ROOT:-/path/to/DIHARD3}            # contains third_dihard_challenge_{dev,eval}
MUSAN=${MUSAN:-/path/to/musan}
RIRS=${RIRS:-/path/to/RIRS_NOISES/simulated_rirs}
SAVE_PATH=${SAVE_PATH:-exps/dihard3_ecapa_8spk}
INIT_MODEL=${INIT_MODEL:-}                              # e.g. a model pretrained on simulated data (fine-tuning)
MAX_SPEAKER=${MAX_SPEAKER:-8}                           # must match prepare_dihard.sh

DEV="$DIHARD_ROOT/third_dihard_challenge_dev/data"
EVAL="$DIHARD_ROOT/third_dihard_challenge_eval/data"

python main.py \
--train_list "$DEV/ts_dev.json" \
--eval_list "$EVAL/ts_eval.json" \
--train_path "$DEV" \
--eval_path "$EVAL" \
--ref_rttm "$EVAL/rttm/all.rttm" \
--musan_path "$MUSAN" \
--rir_path "$RIRS" \
--save_path "$SAVE_PATH" \
${INIT_MODEL:+--init_model "$INIT_MODEL"} \
--max_speaker "$MAX_SPEAKER" \
--warm_up_epoch 5 \
--batch_size 40 \
--rs_len 4 \
--test_shift 4 \
--lr 0.0001 \
--lr_decay 0.95 \
--test_step 1 \
--max_epoch 60 \
--n_cpu 16 \
--train
