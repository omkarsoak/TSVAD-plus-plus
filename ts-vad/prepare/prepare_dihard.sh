#!/usr/bin/env bash
# Build target_audio/, target_embedding/ and ts_{dev,eval}.json for DIHARD3, then split dev into
# train / validation (ts_train.json, ts_val.json, val_all.rttm).
# dev  : data/rttm must contain the GROUND-TRUTH rttm files
# eval : data/rttm must contain your CLUSTERING (e.g. VBx) rttm files (+ all.rttm ground truth, which is skipped)
set -euo pipefail
cd "$(dirname "$0")"

DIHARD_ROOT=${DIHARD_ROOT:-/path/to/DIHARD3}
SOURCE=${SOURCE:-../pretrained_models/ecapa-tdnn.model}
MAX_SPEAKER=${MAX_SPEAKER:-8}
MIN_CLEAN_DUR=${MIN_CLEAN_DUR:-2.0}     # s of non-overlapped speech for a clean profile; else overlap is included
MIN_SPEAKER_DUR=${MIN_SPEAKER_DUR:-0.5} # speakers with less audio than this are dropped
VAL_FRAC=${VAL_FRAC:-0.2}
SEED=${SEED:-0}

for TYPE in ${TYPES:-dev eval}; do
    python prepare_dihard.py \
        --data_path "$DIHARD_ROOT" \
        --type "$TYPE" \
        --max_speaker "$MAX_SPEAKER" \
        --min_clean_dur "$MIN_CLEAN_DUR" \
        --min_speaker_dur "$MIN_SPEAKER_DUR" \
        --source "$SOURCE"
    if [ "$TYPE" = dev ]; then
        python split_dev.py --dev_path "$DIHARD_ROOT/third_dihard_challenge_dev/data" --val_frac "$VAL_FRAC" --seed "$SEED"
    fi
done
