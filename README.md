# TS-VAD++: Overlap-Aware Speaker Diarization

TS-VAD++ is a transformer based speaker diarization system that detects "who spoke when" in multi-speaker audio, including overlapped speech. It takes raw speech, encoded by the first six layers of WavLM, and one ECAPA-TDNN speaker profile per target speaker. A speaker detection transformer processes each speaker's stream, a combining transformer merges the streams, and a per-frame head gives one speech probability per speaker. Post-processing turns these into an RTTM file.

<p align="center">
  <img src="tsvad_plus_plus_architecture.pdf" alt="TS-VAD++ Model Architecture" width="400"/>
</p>

## Differences from the original TS-VAD (2020)

The original TS-VAD [1] computes MFCC features, passes them through a 4 layer CNN, and runs a shared 2 layer BLSTMP speaker detection block with one i-vector per target speaker. A 1 layer BLSTMP combines the speaker streams into 4 fixed outputs. TS-VAD++ keeps this two stage layout and changes the parts around it:

- Features: WavLM (first 6 of 12 layers) replaces MFCC plus CNN.
- Speaker profiles: ECAPA-TDNN embeddings (192 dims, 6 s windows) replace i-vectors. Speakers with little speech still get a profile.
- Sequence modelling: transformer layers replace the BLSTMP layers.
- Output head: a per-frame linear layer for all speakers (see "Output head fix"). The speaker count is `--max_speaker` (default 8), not a fixed 4.
- Protocol: MUSAN and RIR augmentation, a held-out validation split, and checkpoint averaging and post-processing chosen on validation only. The test set is scored once.

## Results on VoxConverse

VoxConverse dev was split 80/20 by recording into train and validation. The test set (232 recordings, 43.5 h) was scored once, after averaging the 5 best checkpoints and tuning the post-processing on validation. Test profiles come from ground truth RTTMs (oracle profiles), so a pipeline with clustered profiles would score worse. Each model was trained once.

| Model | DER collar 0.25 (%) | DER collar 0 (%) | Missed (%) | False alarm (%) | Confusion (%) |
|---|---|---|---|---|---|
| TS-VAD++ (per-frame head) | 10.97 | 15.05 | 5.62 | 3.75 | 5.69 |
| Legacy output head (pre-fix reshape) | 12.17 | 16.68 | 5.58 | 4.27 | 6.84 |

The legacy row is the same model with the old output layer (same features, profiles, data and schedule). It is not a re-implementation of the 2020 TS-VAD, and no run of that model is reported. The 1.6 point gap at collar 0 comes mostly from speaker confusion. With one run per model and validation DER moving by 1 to 2 points between epochs, treat it as suggestive.

Planned, not yet implemented: speaker-wise attention so the output no longer depends on slot order, two pass profile refinement, clustered profiles during training, all 12 WavLM layers, longer chunks.

## Setup

```bash
git clone https://github.com/omkarsoak/TSVAD-plus-plus.git && cd TSVAD-plus-plus
conda env create --name tsvad --file=tsvad.yml     # Python 3.8, torch 2.0.1, CUDA 11.7
conda activate tsvad
pip install sox huggingface_hub pyarrow
cd ts-vad && unzip tools.zip && mkdir -p pretrained_models && cd ..   # tools/ holds the md-eval scorer
```

Needs Linux, conda, `perl` and `sox`. An NVIDIA GPU is expected for real training (the code falls back to CPU for tiny tests).

- Pretrained weights, saved in `ts-vad/pretrained_models/`: [WavLM-Base+.pt](https://drive.google.com/file/d/1-zlAj2SyVJVsbhifwpTlAfrgc9qu-HDb/view?usp=share_link) and [ecapa-tdnn.model](https://drive.google.com/file/d/1E-ju12Jy1fID2l4x-nj0zB5XUHSKWsRB/view?usp=drive_link).
- Optional augmentation data: [MUSAN](https://www.openslr.org/17/) and [RIRS_NOISES](https://www.openslr.org/28/). Missing augmentation types are skipped.

## Data: VoxConverse

VoxConverse [2] is free to download, which makes it a free alternative to DIHARD III (the dataset these scripts were first written for, which needs an LDC licence). `prepare_voxconverse.py` converts the Hugging Face release into the folder layout the scripts expect (`third_dihard_challenge_dev/data` and `third_dihard_challenge_eval/data`), with VoxConverse dev as "dev" and test as "eval". For that reason the root variable is still called `DIHARD_ROOT`.

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('diarizers-community/voxconverse', repo_type='dataset', local_dir='/abs/path/voxconverse_hf')"
cd ts-vad/prepare
python prepare_voxconverse.py --hf_dir /abs/path/voxconverse_hf --out_root /abs/path/voxconverse   # prints speech lost per --max_speaker
DIHARD_ROOT=/abs/path/voxconverse MAX_SPEAKER=8 SOURCE=../pretrained_models/ecapa-tdnn.model ./prepare_dihard.sh
```

Each split gets `wav/`, `rttm/`, `target_audio/`, `target_embedding/` (ECAPA profiles, 6 s window, 1 s step) and `ts_{dev,eval}.json` (40 ms frame labels). Dev also gets `ts_train.json`, `ts_val.json`, `val_recordings.txt` and `val_all.rttm`.

- The "eval" RTTMs are ground truth, so test profiles are oracle profiles. Replace them with clustering output (keeping `all.rttm` as the reference) to evaluate with realistic profiles.
- The top `--max_speaker` speakers by talk time are kept (8 loses 0.87% of dev and 1.4% of test speaker talk time). Use the same value for preparation and training.
- A profile uses a speaker's non-overlapped speech when at least 2 s exists, otherwise all of their speech. Speakers with under 0.5 s are dropped.
- There are no domain labels, so the train/validation split is random and per-domain tables are skipped.

## Training

```bash
cd ts-vad      # run inside tmux so it survives a closed SSH session
CUDA_VISIBLE_DEVICES=0 DIHARD_ROOT=/abs/path/voxconverse MUSAN=/abs/path/musan \
RIRS=/abs/path/RIRS_NOISES/simulated_rirs SAVE_PATH=exps/voxconverse_8spk \
MAX_SPEAKER=8 MAX_EPOCH=40 BATCH_SIZE=40 N_CPU=16 ./run_train.sh
```

It trains on `ts_train.json` and validates on `ts_val.json` after every epoch (about 3 min per epoch on an RTX 6000 Ada). The lowest validation DER is kept as `SAVE_PATH/model/best.model`. A restarted job resumes from the latest checkpoint, so use a fresh `SAVE_PATH` per experiment. Other overrides: `WARM_UP_EPOCH`, `LR`, `LR_DECAY`, `INIT_MODEL`, `LEGACY_HEAD`. `job.pbs` and `job.slurm` wrap the same script for PBS or Slurm clusters.

```
SAVE_PATH/
  model/model_XXXX.model, model/best.model, best.json
  score.txt      per-epoch train loss and validation DER, MS, FA, SC at collars 0.25 and 0
  probs.npz, res_rttm   latest evaluation
```

## Evaluation

```bash
cd ts-vad
DIHARD_ROOT=/abs/path/voxconverse TRAIN_DIR=exps/voxconverse_8spk ./run_final.sh
```

`run_final.sh` makes every choice on validation data: (1) it averages the 5 checkpoints with the lowest validation DER, (2) runs validation inference with a 0.5 s window shift, (3) tunes threshold, median filter, minimum silence and minimum speech on validation (336 settings), and (4) scores the test set once with those settings. Results are in `TRAIN_DIR/final/` (`val/`, `postproc.json`, and `test/score.txt` for the number to report).

DER is reported at collar 0.25 s and 0 s. Any RTTM can be scored with `python scoring_utils.py --ref <ref.rttm> --sys <sys.rttm>`. RTTM segments used to start 40 ms early (an off-by-one frame in the writer); this is fixed in `postprocess.py`.

## Output head fix

The old head reshaped the transformer output `(B, T, 384)` to `(B, max_speaker, T, -1)`, which mixes the time and speaker axes, and was hard-coded for 2 speakers. The head is now a per-frame `Linear(384, max_speaker)`. Checkpoints trained with the old head can be evaluated with `--legacy_head true` (`LEGACY_HEAD=true ./run_final.sh`, and the same variable for `run_train.sh` to train the old head as a baseline).

## Optional: pre-training on simulated data

`ts-vad/prepare/simulated_data_prep.md` describes building simulated mixtures, and `ts-vad/run_train_sim.sh` pre-trains on them (`--simtrain true`). Fine-tune afterwards with `INIT_MODEL=<checkpoint> ./run_train.sh`.

## Notes

- The ECAPA-TDNN speaker encoder was trained on CnCeleb1+2 and the AliMeeting training set.
- If scripts are not executable after cloning: `chmod +x ts-vad/*.sh ts-vad/prepare/*.sh`.

## References

[1] Medennikov et al., Target-Speaker Voice Activity Detection: a Novel Approach for Multi-Speaker Diarization in a Dinner Party Scenario, Interspeech 2020.  

[2] Chung, Huh, Nagrani, Afouras, Zisserman, Spot the conversation: speaker diarisation in the wild, Interspeech 2020.
