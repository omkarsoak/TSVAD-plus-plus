#!/usr/bin/env bash
# Train TS-VAD++ on the DIHARD3 dev TRAIN split; validate on the held-out dev split every epoch.
# The eval set is NOT used here (score it once at the end with run_final.sh).
# Edit the paths below (or export them before calling this script).
set -euo pipefail
cd "$(dirname "$0")"

DIHARD_ROOT=${DIHARD_ROOT:-/path/to/DIHARD3}            # contains third_dihard_challenge_{dev,eval}
MUSAN=${MUSAN:-/path/to/musan}
RIRS=${RIRS:-/path/to/RIRS_NOISES/simulated_rirs}
SAVE_PATH=${SAVE_PATH:-exps/dihard3_ecapa_8spk}
INIT_MODEL=${INIT_MODEL:-}                              # e.g. a model pretrained on simulated data (fine-tuning)
MAX_SPEAKER=${MAX_SPEAKER:-8}                           # must match prepare_dihard.sh
SPEECH_ENCODER=${SPEECH_ENCODER:-pretrained_models/WavLM-Base+.pt}
BATCH_SIZE=${BATCH_SIZE:-40}
MAX_EPOCH=${MAX_EPOCH:-60}
WARM_UP_EPOCH=${WARM_UP_EPOCH:-5}
LR=${LR:-0.0001}
LR_DECAY=${LR_DECAY:-0.95}
N_CPU=${N_CPU:-16}
EVAL_EPOCHS=${EVAL_EPOCHS:-all}                         # validate every epoch (feeds best.model / checkpoint averaging)
LEGACY_HEAD=${LEGACY_HEAD:-false}                       # true = old output reshape (ablation baseline for the head fix)

DEV="$DIHARD_ROOT/third_dihard_challenge_dev/data"

python main.py \
--train_list "$DEV/ts_train.json" \
--train_path "$DEV" \
--eval_list "$DEV/ts_val.json" \
--eval_path "$DEV" \
--ref_rttm "$DEV/val_all.rttm" \
--uem_dir "$DEV/uem_scoring/full" \
--musan_path "$MUSAN" \
--rir_path "$RIRS" \
--save_path "$SAVE_PATH" \
--speech_encoder_pretrain "$SPEECH_ENCODER" \
${INIT_MODEL:+--init_model "$INIT_MODEL"} \
--max_speaker "$MAX_SPEAKER" \
--warm_up_epoch "$WARM_UP_EPOCH" \
--batch_size "$BATCH_SIZE" \
--rs_len 4 \
--test_shift 4 \
--lr "$LR" \
--lr_decay "$LR_DECAY" \
--test_step 1 \
--max_epoch "$MAX_EPOCH" \
--n_cpu "$N_CPU" \
--eval_epochs "$EVAL_EPOCHS" \
--legacy_head "$LEGACY_HEAD" \
--train
