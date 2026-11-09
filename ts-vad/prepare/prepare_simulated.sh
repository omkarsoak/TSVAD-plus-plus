#!/usr/bin/env bash
# Prepare each simulated-data folder (step 3 of the simulated-data pipeline).
set -euo pipefail
cd "$(dirname "$0")"

SIM_DATA=${SIM_DATA:-/path/to/simu3/data}               # contains swb_sre_* folders
SOURCE=${SOURCE:-../pretrained_models/ecapa-tdnn.model}

# Folders to prepare
list=('swb_sre_cv_ns3_beta5_200' 'swb_sre_cv_ns4_beta9_200' 'swb_sre_cv_ns5_beta14_200' 'swb_sre_cv_ns6_beta20_200' 'swb_sre_tr_ns1_beta2_1000' 'swb_sre_tr_ns2_beta2_1000' 'swb_sre_tr_ns3_beta5_1000' 'swb_sre_tr_ns4_beta9_1000' 'swb_sre_tr_ns5_beta14_1000' 'swb_sre_tr_ns6_beta20_1000')

for i in "${list[@]}"; do
    python prepare_simulated.py \
        --data_path "$SIM_DATA/$i" \
        --type simtrain \
        --source "$SOURCE"
done
