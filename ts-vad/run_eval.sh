#!/usr/bin/env bash
# Evaluate one checkpoint on DIHARD3 eval (quick check). For the reported number use run_final.sh,
# which tunes post-processing on validation first.
# Writes $SAVE_PATH/res_rttm, $SAVE_PATH/probs.npz and prints DER at collars 0.25 and 0.00 (+ per domain).
set -euo pipefail
cd "$(dirname "$0")"

DIHARD_ROOT=${DIHARD_ROOT:-/path/to/DIHARD3}
SAVE_PATH=${SAVE_PATH:-exps/dihard3_eval}
MODEL=${MODEL:?set MODEL=/path/to/model_XXXX.model}
MAX_SPEAKER=${MAX_SPEAKER:-8}
LEGACY_HEAD=${LEGACY_HEAD:-false}                       # true only for checkpoints trained before the head fix
POSTPROC_JSON=${POSTPROC_JSON:-}                        # optional output of tune_postproc.py
SPEECH_ENCODER=${SPEECH_ENCODER:-pretrained_models/WavLM-Base+.pt}

EVAL="$DIHARD_ROOT/third_dihard_challenge_eval/data"

python main.py \
--eval_list "$EVAL/ts_eval.json" \
--eval_path "$EVAL" \
--ref_rttm "$EVAL/rttm/all.rttm" \
--uem_dir "$EVAL/uem_scoring/full" \
--save_path "$SAVE_PATH" \
--speech_encoder_pretrain "$SPEECH_ENCODER" \
--rs_len 4 \
--test_shift 0.5 \
--min_silence 0.32 \
--min_speech 0.00 \
--threshold 0.50 \
${POSTPROC_JSON:+--postproc_json "$POSTPROC_JSON"} \
--n_cpu 16 \
--max_speaker "$MAX_SPEAKER" \
--legacy_head "$LEGACY_HEAD" \
--eval \
--init_model "$MODEL"
