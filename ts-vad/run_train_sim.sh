#!/usr/bin/env bash
# Pre-train TS-VAD+ on simulated data (then fine-tune on DIHARD3 with run_train.sh INIT_MODEL=...).
set -euo pipefail
cd "$(dirname "$0")"

SIM_ROOT=${SIM_ROOT:-/path/to/simu3/data/all_files}     # output of the simulated-data prep steps
DIHARD_ROOT=${DIHARD_ROOT:-/path/to/DIHARD3}
MUSAN=${MUSAN:-/path/to/musan}
RIRS=${RIRS:-/path/to/RIRS_NOISES/simulated_rirs}
SAVE_PATH=${SAVE_PATH:-exps/simtrain_ecapa_8spk}
MAX_SPEAKER=${MAX_SPEAKER:-8}

EVAL="$DIHARD_ROOT/third_dihard_challenge_eval/data"

python main.py \
--train_list "$SIM_ROOT/all_simtrain.json" \
--eval_list "$EVAL/ts_eval.json" \
--train_path "$SIM_ROOT" \
--eval_path "$EVAL" \
--ref_rttm "$EVAL/rttm/all.rttm" \
--musan_path "$MUSAN" \
--rir_path "$RIRS" \
--save_path "$SAVE_PATH" \
--max_speaker "$MAX_SPEAKER" \
--warm_up_epoch 10 \
--batch_size 40 \
--rs_len 4 \
--test_shift 4 \
--lr 0.0001 \
--test_step 1 \
--max_epoch 40 \
--n_cpu 16 \
--train \
--simtrain
