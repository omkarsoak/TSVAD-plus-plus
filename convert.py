"""Convert every .flac in a folder to 16 kHz .wav.

usage: python convert.py <flac_dir> <wav_dir> [--resample-8k]
  --resample-8k : down-sample to 8 kHz and back to 16 kHz (matches the simulated pre-training data)
"""
import os, sys, argparse
import sox
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('source_dir')
parser.add_argument('target_dir')
parser.add_argument('--resample-8k', action='store_true')
args = parser.parse_args()

os.makedirs(args.target_dir, exist_ok=True)
for filename in tqdm(sorted(os.listdir(args.source_dir))):
    if not filename.endswith(".flac"):
        continue
    flac_path = os.path.join(args.source_dir, filename)
    wav_path = os.path.join(args.target_dir, os.path.splitext(filename)[0] + ".wav")
    tfm = sox.Transformer()
    if args.resample_8k:
        tmp = wav_path + ".8k.wav"
        tfm.set_output_format(rate=8000)
        tfm.build(flac_path, tmp)
        up = sox.Transformer()
        up.set_output_format(rate=16000)
        up.build(tmp, wav_path)
        os.remove(tmp)
    else:
        tfm.set_output_format(rate=16000)
        tfm.build(flac_path, wav_path)
print("Conversion complete")
