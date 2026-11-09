#!/usr/bin/env bash
# Build target_audio/, target_embedding/ and ts_{dev,eval}.json for DIHARD3.
# dev  : run on data/rttm containing the GROUND-TRUTH rttm files
# eval : run on data/rttm containing your CLUSTERING (e.g. VBx) rttm files (+ all.rttm ground truth, which is skipped)
set -euo pipefail
cd "$(dirname "$0")"

DIHARD_ROOT=${DIHARD_ROOT:-/path/to/DIHARD3}
SOURCE=${SOURCE:-../pretrained_models/ecapa-tdnn.model}
MAX_SPEAKER=${MAX_SPEAKER:-8}

for TYPE in ${TYPES:-dev eval}; do
    python prepare_dihard.py \
        --data_path "$DIHARD_ROOT" \
        --type "$TYPE" \
        --max_speaker "$MAX_SPEAKER" \
        --source "$SOURCE"
done
