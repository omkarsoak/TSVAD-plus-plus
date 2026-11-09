#!/usr/bin/env bash
# Copy the first N dev and eval recordings (wav + rttm) of a full DIHARD3 tree into a small one,
# for quick local tests. The eval rttm files are ground truth (oracle speaker profiles).
# usage: ./make_mini_dihard.sh <full DIHARD3 root> <mini root> [N=5]
set -euo pipefail
SRC=${1:?full DIHARD3 root}; DST=${2:?mini root}; N=${3:-5}
for SPLIT in dev eval; do
    S="$SRC/third_dihard_challenge_$SPLIT/data"; D="$DST/third_dihard_challenge_$SPLIT/data"
    mkdir -p "$D/wav" "$D/rttm"
    for RTTM in $(ls "$S"/rttm/DH_*.rttm | head -n "$N"); do
        ID=$(basename "$RTTM" .rttm)
        cp "$RTTM" "$D/rttm/"
        cp "$S/wav/$ID.wav" "$D/wav/"
    done
    # domain lists for the stratified dev split and per-domain DER (small text files)
    if [ -d "$S/uem_scoring" ]; then cp -r "$S/uem_scoring" "$D/"; fi
done
cat "$DST"/third_dihard_challenge_eval/data/rttm/DH_*.rttm > "$DST/third_dihard_challenge_eval/data/rttm/all.rttm"
echo "Mini DIHARD3 with $N recordings per split written to $DST"
