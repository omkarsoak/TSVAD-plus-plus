"""Split the prepared DIHARD3 dev set into train / validation lists (by recording, stratified by domain).

Run after prepare_dihard.py --type dev. Writes, next to ts_dev.json:
  ts_train.json        training lines (recordings not in validation)
  ts_val.json          validation lines
  val_recordings.txt   validation recording ids with their domain
  val_all.rttm         ground-truth RTTM of the validation recordings (DER reference)

usage: python split_dev.py --dev_path <DIHARD3>/third_dihard_challenge_dev/data [--val_frac 0.2] [--seed 0]

Domains are read from <dev_path>/uem_scoring/{full,core}/<domain>.uem (DIHARD release layout). If those
files are missing, the split is random over all recordings. Use the validation list for choosing epochs,
checkpoint averaging and post-processing; score the eval set only once at the end.
"""
import argparse, glob, json, os, random
from collections import defaultdict

def domains_from_uem(dev_path):
	for sub in ('full', 'core'):
		files = sorted(glob.glob(os.path.join(dev_path, 'uem_scoring', sub, '*.uem')))
		files = [f for f in files if os.path.splitext(os.path.basename(f))[0].lower() != 'all']
		if files:
			rec2dom = {}
			for f in files:
				dom = os.path.splitext(os.path.basename(f))[0]
				for line in open(f):
					if line.strip():
						rec2dom[line.split()[0]] = dom
			return rec2dom, os.path.join(dev_path, 'uem_scoring', sub)
	return {}, None

def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--dev_path', required=True)
	ap.add_argument('--val_frac', type=float, default=0.2)
	ap.add_argument('--seed', type=int, default=0)
	a = ap.parse_args()

	src = os.path.join(a.dev_path, 'ts_dev.json')
	lines = [l for l in open(src).read().splitlines() if l.strip()]
	recs = sorted(set(json.loads(l)['filename'] for l in lines))
	if len(recs) < 3:
		raise SystemExit('Need at least 3 prepared dev recordings (2 for training, 1 for validation); found %d' % len(recs))

	rec2dom, uem_dir = domains_from_uem(a.dev_path)
	groups = defaultdict(list)
	for r in recs:
		groups[rec2dom.get(r, 'all')].append(r)
	if uem_dir:
		print('Stratifying by %d domains from %s' % (len(groups), uem_dir))
	else:
		print('No uem_scoring/*.uem found: random split over all recordings')

	rng = random.Random(a.seed)
	val = set()
	for dom in sorted(groups):
		g = sorted(groups[dom]); rng.shuffle(g)
		k = int(round(a.val_frac * len(g)))
		if len(g) >= 2:
			k = max(k, 1)  # every domain with >= 2 recordings appears in validation
		val.update(g[:k])
	if len(recs) - len(val) < 2:  # train needs >= 2 recordings (filler speaker slots come from other recordings)
		for r in sorted(val)[:2 - (len(recs) - len(val))]:
			val.discard(r)
	train = [r for r in recs if r not in val]

	with open(os.path.join(a.dev_path, 'ts_train.json'), 'w') as ft, open(os.path.join(a.dev_path, 'ts_val.json'), 'w') as fv:
		for l in lines:
			(fv if json.loads(l)['filename'] in val else ft).write(l + '\n')
	with open(os.path.join(a.dev_path, 'val_recordings.txt'), 'w') as f:
		for r in sorted(val):
			f.write('%s %s\n' % (r, rec2dom.get(r, 'all')))

	# ground-truth reference for the validation recordings (dev rttm/ holds ground truth)
	n_seg = 0
	with open(os.path.join(a.dev_path, 'val_all.rttm'), 'w') as f:
		for rttm in sorted(glob.glob(os.path.join(a.dev_path, 'rttm', '*.rttm'))):
			if os.path.basename(rttm) == 'all.rttm':
				continue
			for line in open(rttm):
				p = line.split()
				if len(p) >= 8 and p[0] == 'SPEAKER' and p[1] in val:
					f.write(line); n_seg += 1

	print('Recordings: %d train, %d validation (seed %d, val_frac %.2f)' % (len(train), len(val), a.seed, a.val_frac))
	for dom in sorted(groups):
		nv = sum(1 for r in groups[dom] if r in val)
		print('  %-22s %3d train  %3d val' % (dom, len(groups[dom]) - nv, nv))
	print('Wrote ts_train.json, ts_val.json, val_recordings.txt, val_all.rttm (%d segments) in %s' % (n_seg, a.dev_path))

if __name__ == '__main__':
	main()
