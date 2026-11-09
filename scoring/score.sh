#!/usr/bin/env bash
# Official-style DIHARD3 scoring (per-domain, CORE set, collar 0) using the uem_scoring files.
# usage: ./score.sh <system_rttm> [reference_rttm]
#   DIHARD_EVAL_DIR must point to third_dihard_challenge_eval
set -euo pipefail
cd "$(dirname "$0")"
DIHARD_EVAL_DIR=${DIHARD_EVAL_DIR:?set DIHARD_EVAL_DIR=/path/to/DIHARD3/third_dihard_challenge_eval}
SYS_RTTM=${1:-./res.rttm}
REF_RTTM=${2:-./all_rttm}

./score_dihard.sh "$DIHARD_EVAL_DIR" "$REF_RTTM" "$SYS_RTTM"
