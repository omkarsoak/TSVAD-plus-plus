"""Convert the Hugging Face VoxConverse dataset (diarizers-community/voxconverse parquet) into the DIHARD3 folder layout
that prepare_dihard.py / split_dev.py / the training scripts expect.

  <out_root>/third_dihard_challenge_dev/data/{wav,rttm}            <- VoxConverse dev
  <out_root>/third_dihard_challenge_eval/data/{wav,rttm,all.rttm}  <- VoxConverse test

The "eval" rttm folder holds the ground truth (oracle speaker profiles) because no diarizer is available to produce
clustering output yet; replace those files with clustering RTTMs later and keep all.rttm as the reference.
No uem_scoring is written (VoxConverse has no domains), so per-domain tables are skipped.

usage: python prepare_voxconverse.py --hf_dir /path/voxconverse_hf --out_root /path/voxconverse [--limit N]
"""
import argparse, glob, io, os
from collections import defaultdict
import numpy, soundfile
import pyarrow.parquet as pq

CAPS = (4, 6, 8, 10, 12, 16)

def convert(hf_dir, out_root, hf_split, dihard_type, limit, stats):
	data = os.path.join(out_root, 'third_dihard_challenge_%s' % dihard_type, 'data')
	wav_dir, rttm_dir = os.path.join(data, 'wav'), os.path.join(data, 'rttm')
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(rttm_dir, exist_ok=True)
	all_lines, n, hours = [], 0, 0.0
	for shard in sorted(glob.glob(os.path.join(hf_dir, 'data', hf_split + '-*.parquet'))):
		table = pq.read_table(shard)
		for i in range(table.num_rows):
			if limit and n >= limit:
				break
			row = {c: table.column(c)[i].as_py() for c in table.column_names}
			rec = os.path.splitext(os.path.basename(row['audio']['path']))[0]
			audio, sr = soundfile.read(io.BytesIO(row['audio']['bytes']), dtype='int16')
			if audio.ndim > 1:
				audio = audio[:, 0]
			assert sr == 16000, '%s has sample rate %d' % (rec, sr)
			soundfile.write(os.path.join(wav_dir, rec + '.wav'), audio, sr, subtype='PCM_16')
			hours += len(audio) / sr / 3600
			segs = sorted(zip(row['timestamps_start'], row['timestamps_end'], row['speakers']))
			talk = defaultdict(float)
			lines = []
			for s, e, spk in segs:
				if e <= s:
					continue
				spk = str(spk).replace('_', '')  # prepare_dihard.py reads the part after the last '_' as a name
				talk[spk] += e - s
				lines.append('SPEAKER %s 1 %.3f %.3f <NA> <NA> %s <NA> <NA>' % (rec, s, e - s, spk))
			with open(os.path.join(rttm_dir, rec + '.rttm'), 'w') as f:
				f.write('\n'.join(lines) + '\n')
			all_lines += lines
			order = sorted(talk.values(), reverse=True)
			total = sum(order)
			for cap in CAPS:
				stats[cap][0] += sum(order[cap:])
				stats[cap][1] += total
			n += 1
	if dihard_type == 'eval':
		with open(os.path.join(rttm_dir, 'all.rttm'), 'w') as f:
			f.write('\n'.join(all_lines) + '\n')
	print('%s -> %s: %d recordings, %.1f h' % (hf_split, data, n, hours))

def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--hf_dir', required=True)
	ap.add_argument('--out_root', required=True)
	ap.add_argument('--limit', type=int, default=0, help='convert only the first N recordings of each split (quick tests)')
	a = ap.parse_args()
	for hf_split, dihard_type in (('dev', 'dev'), ('test', 'eval')):
		stats = {cap: [0.0, 0.0] for cap in CAPS}
		convert(a.hf_dir, a.out_root, hf_split, dihard_type, a.limit, stats)
		print('  speaker talk time lost if only the top-K speakers are kept:')
		for cap in CAPS:
			lost, total = stats[cap]
			print('    K=%-2d  %5.2f%%' % (cap, 100 * lost / max(total, 1e-9)))

if __name__ == '__main__':
	main()
