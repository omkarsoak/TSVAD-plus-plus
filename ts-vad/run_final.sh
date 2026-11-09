#!/usr/bin/env bash
# Final, test-set-clean evaluation of a training run:
#   1. average the AVG_N checkpoints with the best validation DER (or use MODEL=... directly)
#   2. run the model on the validation split (0.5 s shift) and save per-frame probabilities
#   3. tune post-processing (threshold, median filter, min silence/speech) on validation only
#   4. score the DIHARD3 eval set ONCE with those settings (overall + per-domain DER)
set -euo pipefail
cd "$(dirname "$0")"

DIHARD_ROOT=${DIHARD_ROOT:-/path/to/DIHARD3}
TRAIN_DIR=${TRAIN_DIR:-exps/dihard3_ecapa_8spk}          # SAVE_PATH used by run_train.sh
OUT=${OUT:-$TRAIN_DIR/final}
AVG_N=${AVG_N:-5}
MODEL=${MODEL:-}                                        # set to skip averaging and use one checkpoint
MAX_SPEAKER=${MAX_SPEAKER:-8}
LEGACY_HEAD=${LEGACY_HEAD:-false}
SPEECH_ENCODER=${SPEECH_ENCODER:-pretrained_models/WavLM-Base+.pt}
N_CPU=${N_CPU:-16}
BATCH_SIZE=${BATCH_SIZE:-40}
TEST_SHIFT=${TEST_SHIFT:-0.5}

DEV="$DIHARD_ROOT/third_dihard_challenge_dev/data"
EVAL="$DIHARD_ROOT/third_dihard_challenge_eval/data"
mkdir -p "$OUT"

if [ -z "$MODEL" ]; then
    MODEL="$OUT/avg_best${AVG_N}.model"
    python average_checkpoints.py --save_path "$TRAIN_DIR" --best "$AVG_N" --out "$MODEL"
fi

common=(--rs_len 4 --test_shift "$TEST_SHIFT" --n_cpu "$N_CPU" --batch_size "$BATCH_SIZE" --max_speaker "$MAX_SPEAKER"
        --legacy_head "$LEGACY_HEAD" --speech_encoder_pretrain "$SPEECH_ENCODER" --init_model "$MODEL" --eval)

echo "== [1/3] validation inference"
python main.py "${common[@]}" \
    --eval_list "$DEV/ts_val.json" --eval_path "$DEV" --ref_rttm "$DEV/val_all.rttm" \
    --uem_dir "$DEV/uem_scoring/full" --save_path "$OUT/val"

echo "== [2/3] tune post-processing on validation"
python tune_postproc.py --probs "$OUT/val/probs.npz" --ref "$DEV/val_all.rttm" --out "$OUT/postproc.json" --n_jobs "$N_CPU"

echo "== [3/3] eval set (scored once)"
python main.py "${common[@]}" \
    --eval_list "$EVAL/ts_eval.json" --eval_path "$EVAL" --ref_rttm "$EVAL/rttm/all.rttm" \
    --uem_dir "$EVAL/uem_scoring/full" --postproc_json "$OUT/postproc.json" --save_path "$OUT/test"

echo "Done. Validation: $OUT/val/score.txt   Eval: $OUT/test/score.txt   RTTM: $OUT/test/res_rttm"
