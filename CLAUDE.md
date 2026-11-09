# CLAUDE.md — TS-VAD++ (speaker diarization)

Context for working in this repo. README.md is the user-facing guide, `explaination.md` (gitignored) is a beginner explainer, and this file records architecture, data contracts, decisions, gotchas and open work.

**Project name:** the user's project is **TS-VAD++: Overlap-Aware Speaker Diarization**, built on the TS-VAD+ code in this repo. Don't call it "multimodal": it is audio-only. The user rejected that wording for accuracy.

## Status (2026-11-09)

- **Done:** Step 0 (evaluation protocol) and Tier 1 (cheap wins) of the improvement plan below, verified end to end on CPU with synthetic data.
- **Next:** the user moves the project to their GPU cluster and trains the new baseline (fixed head + validation protocol + short-speaker profiles).
- **Diagrams (done, not compiled):** `diagrams/workflow.md` (data to final evaluation pipeline) and `diagrams/architecture.md` (model data flow) are Obsidian TikZJax blocks written to the user's TikZ rules. They are untested because LaTeX is not installed locally (the user declined installing it), so ask the user to confirm they render and fix any TikZJax errors they report.
- **User's global instructions live on the Windows side:** `/mnt/c/Users/omkar/.claude/CLAUDE.md` (the WSL `~/.claude/CLAUDE.md` is empty). Read it when the user mentions their main instructions. Key rules: banned words (Delve, Tapestry, Vibrant, Landscape, Realm, Embark, Excels, Vital, Pivotal, Thrilled); no dashes of any kind in sentences, and no bold or italics in text given to the user; `$ $` and `$$ $$` for math; TikZ for TikZJax must start with `\begin{document}`, use `\begin{tikzpicture}[font=\sffamily\bfseries, >=latex]`, no `\textbf`, a `\useasboundingbox` padding, simple diagrams, basic colors. Also saved as a memory note.
- **Not committed:** git is initialised with no commits yet; everything is staged or untracked.

## Environment

- **Local machine (WSL2):** Intel Iris Xe GPU only (no CUDA), 16 cores, ~7 GB RAM. Use it for editing and CPU smoke tests only, never real training.
- **Conda env `tsvad`** (created 2026-11-09 from `tsvad.yml`): Python 3.8.17, torch 2.0.1+cu117, torchaudio 2.0.2, numpy 1.22.4, plus `pip install sox`. Run scripts with `PATH=/home/omkar/miniforge3/envs/tsvad/bin:$PATH` or `conda activate tsvad`. The command-line `sox` is **not** installed (needs `sudo apt-get install sox`, which the user does).
- **Default env `myenv`:** Python 3.14, no torch.
- **Training:** the user's GPU cluster (scheduler unknown; `ts-vad/job.pbs` and `ts-vad/job.slurm` both exist). The original runs used 1 GPU, 16 CPUs, 128–189 GB RAM and a 24 h walltime.
- **CPU mode:** the trainer picks CUDA if available, otherwise CPU (AMP off, `map_location` on load).
- **`ts-vad/tools/`** is unzipped from `tools.zip` (gitignored). It contains `tools.py` (init_system + the original post-processing helpers) and SCTK 2.4.12. The md-eval copy inside SCTK is **modified** (prints a short `DER/MS/FA/SC` line) and is no longer used for scoring. `tools/SCTK-2.4.12/all.rttm` is a leftover AliMeeting reference (`R8001_M8004_MS801`), unused.
- **`.gitignore`:** `__pycache__/`, `*.py[cod]`, `resume.md`, `explaination.md`, `ts-vad/tools/`. Suggested to the user but not yet added: `ts-vad/exps/`, `ts-vad/pretrained_models/`, `*.model`.

## Layout

```
convert.py                      FLAC -> 16 kHz WAV (optional --resample-8k round-trip); args: <flac_dir> <wav_dir>
tsvad.yml                       conda env
scoring/                        DIHARD3 per-domain CORE scoring (score.sh -> score_dihard.sh -> utils/md-eval.pl + rttm_from_uem.py)
  utils/md-eval.pl              UNMODIFIED md-eval, used by ts-vad/scoring_utils.py
  res.rttm, all_rttm            bundled example system output / ground truth (DIHARD3 eval, 259 recordings)
diagrams/                       workflow.md, architecture.md (TikZJax blocks for Obsidian)
ts-vad/
  main.py                       entry point (argparse; --train / --eval); best-model tracking
  trainer.py                    train_network / eval_network (inference, probs.npz, RTTM, DER), checkpoint load/save, resume
  loss.py                       output head (fc) + BCE loss
  dataLoader.py                 train_loader (random chunks + MUSAN/RIR aug) / eval_loader (sliding windows)
  postprocess.py                probs -> binary -> RTTM (median filter, threshold, min_silence, min_speech); 40 ms fix
  scoring_utils.py              md-eval wrapper (DER/MS/FA/SC from seconds) + per-domain DER via uem_scoring; CLI
  tune_postproc.py              grid-search post-processing on validation probs.npz -> postproc.json
  average_checkpoints.py        average weights of best-N (by validation DER in score.txt) or last-N checkpoints
  run_train.sh                  train on ts_train.json, validate on ts_val.json every epoch
  run_final.sh                  average best-N -> val inference -> tune -> eval set scored ONCE
  run_eval.sh                   quick eval of one checkpoint on the eval set (optional POSTPROC_JSON)
  run_train_sim.sh              simulated-data pre-training (validates on the dev val split)
  run_smoke.sh                  run_train.sh + run_final.sh with tiny settings (CPU)
  job.pbs / job.slurm / prepare/*.pbs   cluster wrappers (activate `tsvad`, call the run scripts)
  job_copy.pbs                  dead: calls nonexistent test_ddp.py
  model/ts_vad.py               TS_VAD model
  model/WavLM.py, modules.py    WavLM implementation + attention modules
  model/selective_scan_interface.py, model/ecapa_tdnn.py   unused
  prepare/prepare_dihard.py(.sh)     targets, embeddings, ts_{split}.json; the .sh also runs split_dev.py after dev
  prepare/split_dev.py               dev -> ts_train.json / ts_val.json / val_recordings.txt / val_all.rttm
  prepare/make_mini_dihard.sh        copy N recordings per split (+ uem_scoring) for local tests
  prepare/prepare_dihard_old.py      older ECAPA-only version, unused
  prepare/prepare_simulated.py(.sh), prepare/simulated_data_prep_utils/   simulated data prep
  prepare/speaker_encoder.py         ECAPA_TDNN used by prep (raw 16 kHz wav -> 192-d)
  tools.zip                     must be unzipped in ts-vad/
  pretrained_models/            (not in repo) WavLM-Base+.pt, ecapa-tdnn.model
```

## Model (ts-vad/model/ts_vad.py + loss.py)

- Frames are 40 ms (25 per second). Chunks are `rs_len` seconds (scripts use 4 s, so T = 100). Audio is read as `start*640 : stop*640+240` samples.
- `rs_forward`: WavLM (first **6** of 12 encoder layers, `cfg.encoder_layers = 6`, `strict=False`) -> 768-d at 20 ms -> Conv1d stride 2 -> **(B, T, 192)**.
- `ts_forward`: identity on precomputed ECAPA embeddings **(B, max_speaker, 192)**.
- `cat_forward`:
  1. For each speaker, concatenate the embedding with the speech features (384-d), then a positional encoding and the 3-layer "single" transformer.
  2. Stack all speakers -> (B, spk·384, T), then `backend_down` Conv1d -> 384. This step is slot-order dependent and squeezes all speakers into 384 dims; see Tier 2.1.
  3. A 3-layer "multi" transformer -> **(B, T, 384)**.
- **Head (fixed 2026-11-08):** `Loss.fc = Linear(384, max_speaker)` per frame -> (B, spk, T) logits; BCE-with-logits averaged over speakers.
- **Legacy head (`--legacy_head true`):** the original raw `reshape(B, spk, T, 384/spk)` + `Linear(384/spk, 1)`. It mixes the time and speaker axes; verified on CPU, input frame 99 only influenced speaker 7, frames 92–99. Keep it only for evaluating old checkpoints.
- WavLM is frozen while `epoch < warm_up_epoch`. AdamW, StepLR(test_step, lr_decay), AMP on CUDA only.
- Checkpoints are a single OrderedDict (ts_vad + ts_loss keys, no prefix). `load_parameters` remaps names and skips shape mismatches (with a `--legacy_head` hint for `fc`).

## Data contract

- `ts_{split}.json` is JSON-lines: `{"filename", "speaker_key": "1".."N", "speaker_id": "speakerX", "labels": [0/1 per 40 ms frame over the full recording]}`.
- `target_audio/<rec>/{1..N}.wav` holds the profile audio; `all.wav` is the full mix. `target_embedding/<rec>/{k}.pt` is a (num_windows, 192) tensor (6 s window, 1 s step; **one window over the whole clip if it's shorter than 6 s**). Train picks a random row; eval uses the mean.
- **Speaker selection (new):**
  - The profile uses clean (non-overlapped) speech if there's ≥ `--min_clean_dur` (2 s) of it; otherwise all of the speaker's speech, overlap included (`--overlap_fallback`, default true).
  - Speakers with < `--min_speaker_dur` (0.5 s) of profile audio are dropped.
  - The top `--max_speaker` by **total** talk time are kept, with ids 1..N and no gaps.
  - Per-recording output dirs are wiped before writing.
  - A summary line reports kept/dropped counts and the speech lost to drops.
  - The old code dropped every speaker with < 6 s of clean speech (guaranteed missed speech) and could leave id gaps.
- Empty speaker slots (id 0) get an embedding from a speaker in a **different recording** with all-zero labels (train: cached per recording; prefers speaker names not in this recording). DIHARD reuses `speaker1..N` names in every recording.
- **RTTM roles:**
  - `dev/data/rttm/*.rttm`: ground truth.
  - `eval/data/rttm/DH_EVAL_*.rttm`: clustering output (profiles).
  - `eval/data/rttm/all.rttm`: ground truth for scoring only (skipped by prep).
  - `dev/data/val_all.rttm`: ground truth of the validation recordings (written by split_dev.py, outside `rttm/`).
- **Validation split:** `split_dev.py` holds out ~20% of dev recordings (`--val_frac`, `--seed 0`), stratified by domain from `dev/data/uem_scoring/{full,core}/<domain>.uem` (random split if missing). Every domain with ≥ 2 recordings appears in validation, and train keeps ≥ 2 recordings.
- Validation profiles come from ground truth (oracle), while eval profiles come from clustering, so validation DER < eval DER. Matching eval conditions requires a `devval` split prepared from clustering RTTMs (documented in the README; not automated).
- Speaker label parsing strips a `speaker` prefix if present (VBx `1`, `spk01` also work). The speaker-encoder type comes from the checkpoint file-name prefix (`ecap` default, `camp`, `eres`). The ECAPA loader strips a `speaker_encoder.` prefix and raises if 0 tensors match.
- DIHARD audio goes 16 kHz -> 8 kHz -> 16 kHz to match the simulated pre-training data.

## Eval / post-processing

- `eval_network`:
  - sliding windows (`--test_shift`; 4 s during training, 0.5 s in run_final/run_eval), with **vectorised** per-frame averaging (sum/count arrays);
  - skips filler slots (id 0); the old loop used `range(max id)` and could emit a filler slot as speaker "0";
  - saves `SAVE_PATH/probs.npz` (`{"<rec>-<spk>": float32 per-frame P}`);
  - writes the RTTM via `postprocess.write_rttm`, then scores via `scoring_utils.md_eval` at collars 0.25 and 0;
  - prints a per-domain table if `--uem_dir` exists;
  - returns the collar-0 DER.
- **Post-processing:** median filter (`--median_kernel`, default 21, 1 = off) -> threshold -> fill silences of ≤ int(min_silence // 0.04) frames (edges included) -> drop speech of ≤ int(min_speech // 0.04) frames. Verified identical to the original `tools.py` functions on 300 random cases.
- **40 ms RTTM fix:** the old writer set a segment's start to the index of the preceding silent frame, so every segment (except one at t = 0) started 40 ms early. On the bundled `res.rttm`, shifting by +40 ms gives DER **24.40 -> 23.84%** at collar 0 (11.65 -> 11.61% at 0.25). Fixed in `postprocess.segments`.
- **`scoring_utils.md_eval`** uses the unmodified `scoring/utils/md-eval.pl`. It computes MS/FA/SC from seconds and cross-checks against md-eval's OVERALL line (raises on > 0.01 mismatch). md-eval labels false alarm as `FALARM SPEAKER TIME`, not "FALSE ALARM". Per-domain scoring filters both RTTMs to the domain's recordings that are in the reference, and passes the domain UEM.
- **`tune_postproc.py`:** grid of threshold {0.3..0.6 step 0.05} × median {1, 11, 21, 31} × min_silence {0, .16, .32, .48} × min_speech {0, .08, .16} = 336 settings, multiprocessing, selected at collar 0. Writes `{"best": {...}, "best_scores", "grid", ...}`; `main.py --postproc_json` loads `best`.
- **Best model:** after each validation eval in training, `model/best.model` + `best.json` (epoch, DER) are updated when DER improves. They survive requeues.
- **`average_checkpoints.py`:** `--best N` parses `Eval full 0.00: <epoch> epoch, DER x%` lines in score.txt; `--last N` takes the newest N. Floats are averaged in float64; integer buffers come from the newest model.
- **Eval-only runs** (`--eval`) no longer build or scan the training list.

## Training / resume behaviour

- When training, an existing `SAVE_PATH/model/model_*.model` takes priority over `--init_model` (requeue-safe); in `--eval` mode `--init_model` wins. Use a fresh `SAVE_PATH` per experiment.
- run_train.sh env overrides: `DIHARD_ROOT MUSAN RIRS SAVE_PATH INIT_MODEL MAX_SPEAKER SPEECH_ENCODER BATCH_SIZE MAX_EPOCH WARM_UP_EPOCH LR LR_DECAY N_CPU EVAL_EPOCHS`.
- run_final.sh env: `DIHARD_ROOT TRAIN_DIR OUT AVG_N MODEL MAX_SPEAKER LEGACY_HEAD SPEECH_ENCODER N_CPU BATCH_SIZE TEST_SHIFT`.
- prepare_dihard.sh env: `DIHARD_ROOT SOURCE MAX_SPEAKER MIN_CLEAN_DUR MIN_SPEAKER_DUR VAL_FRAC SEED TYPES`.
- `--simtrain` / `--legacy_head` use `str2bool` with `nargs='?'`. `--max_speaker` defaults to 8 and must match prep.

## Results / baselines

- README table (DIHARD3, collar 0): VBx+OSD 21.97, EEND 20.05, mm-TS-VAD+ 22.36, **TS-VAD+ best 19.11** (initial 22.57), SOTA ANSD-MA-MSE 16.76. Produced with the **legacy head**, with epochs and thresholds selected on the eval set (optimistic). No ablation, and no comparison against the original 2020 TS-VAD.
- Bundled `scoring/res.rttm` vs `scoring/all_rttm` (md-eval, no UEM):
  - collar 0: DER **24.40%** (MS 13.12, FA 7.62, SC 3.65); with the 40 ms fix, 23.84%;
  - collar 0.25: 11.65% (MS 7.33, FA 1.44, SC 2.88).
- **Missed speech is the dominant error.**
- Synthetic end-to-end smoke test (2026-11-09; random encoders, so the numbers are meaningless): the full chain ran in 125 s and 1.4 GB. Tuning moved validation DER from 100% (default threshold 0.5) to 37% (threshold 0.3), which shows why tuning on validation matters for an under-trained model.

## Improvement plan for TS-VAD++

Step 0 and Tier 1 are implemented. Train each step as its own run and record validation DER, which builds the ablation table the original project lacks. Don't promise gains in advance.

- **Step 0 (done):**
  - validation split (`split_dev.py`);
  - validate every epoch during training, with `best.model`;
  - eval set scored once (`run_final.sh`);
  - per-domain DER (`--uem_dir`, `scoring_utils.py`);
  - new baseline with the fixed head: **user to train on the cluster**.
- **Tier 1 (done):**
  - keep short speakers (profile fallback);
  - tune post-processing on validation (`tune_postproc.py`, `--median_kernel`);
  - checkpoint averaging (`average_checkpoints.py`);
  - 40 ms RTTM fix (found while doing this).
  - "Better first guess" (pyannote 3.x / VBx+OSD as input RTTMs) needs no code: compare candidate RTTMs with `scoring_utils.py`. pyannote is not in the env.
- **Tier 2 (next; the main TS-VAD++ contributions):**
  1. **Speaker-wise attention:** keep per-speaker streams (B, spk, T, D), let speakers attend to each other at each frame, and apply one shared per-speaker output layer, replacing `backend_down` + `Linear(384, spk)`. This makes the model slot-order independent and removes the 384-dim bottleneck (as in the NSD-MA-MSE line of work). Headline architectural change.
  2. **Two-pass (iterative) TS-VAD:** re-extract profiles from the first pass's output and run again.
  3. **Realistic profiles in training:** mix in profiles from clustering output on dev, plus embedding noise/dropout (fixes the oracle-train vs clustering-test mismatch).
  4. **All 12 WavLM layers** with a learned weighted sum (or WavLM-Large).
  5. **Longer chunks** (8–16 s; reduce batch size).
  6. **Attention over a speaker's embedding windows** instead of the mean.
- **Tier 3:**
  - DOVER-Lap fusion of several models or seeds;
  - overlap-weighted loss;
  - better simulated conversations;
  - multi-GPU (DDP);
  - a config file instead of long CLIs;
  - remove dead files (`job_copy.pbs`, `prepare_dihard_old.py`, Mamba file) once the user confirms.

## Change log

- **2026-11-08:**
  - head fix + `--legacy_head`;
  - `--max_speaker` default 8;
  - ECAPA support, `all.rttm` skip and robust labels in prep;
  - all NUS/NTU cluster paths removed (env-var scripts, PBS + Slurm jobs);
  - `--simtrain` parsing;
  - `--eval_epochs`, resume-over-init_model;
  - README rewritten.
- **2026-11-09 (a):**
  - CPU support;
  - optional MUSAN/RIR;
  - filler-slot bug in `train_loader.load_ts` (crashed on small sets; skewed on DIHARD);
  - missing `MD_EVAL` constant (scoring would have crashed);
  - `run_smoke.sh`, `make_mini_dihard.sh`.
- **2026-11-09 (b):**
  - Step 0 + Tier 1 (above), the 40 ms fix, the speaker-"0" emission fix, vectorised eval, eval-only skips the train list;
  - `run_train_sim.sh` validates on dev val (not eval);
  - README sections for validation, final evaluation and the timing fix.
  - Verified with the real scripts on a synthetic DIHARD-layout tree (edge cases: 0.3 s speaker dropped, overlap-fallback profile, 6 speakers > max_speaker 4, 2 domains): ids/wav/embedding consistency, no train/val leakage, best.model, averaging, tuning, final eval with per-domain table.
  - Not verified: real ECAPA/WavLM weights, real DIHARD data, GPU/AMP path after these changes.

## Conventions

- Tabs in most `ts-vad/*.py` files (trainer, main, dataLoader, prepare, postprocess, scoring_utils, tune_postproc, average_checkpoints); 4 spaces in model/ and loss.py. Match the file you edit.
- Scripts `cd "$(dirname "$0")"`, so they can be run from anywhere. Python entry points expect `ts-vad/tools/` to exist.
- Never select epochs, thresholds or settings on the eval set; use `ts_val.json` / `val_all.rttm`.
- Test changes with the synthetic DIHARD-layout generator approach: random-init ECAPA saved as `ecapa_*.model`, plus a fake WavLM `{'cfg': {}, 'model': {}}`. Run the real scripts with `SOURCE=` / `SPEECH_ENCODER=` pointing at them.
