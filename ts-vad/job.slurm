#!/bin/bash
# Slurm job: train TS-VAD+.  Submit from ts-vad/:
#   sbatch --export=ALL,DIHARD_ROOT=...,MUSAN=...,RIRS=... job.slurm
#SBATCH --job-name=tsvad_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=23:59:00
#SBATCH --output=tsvad_train_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tsvad
./run_train.sh
