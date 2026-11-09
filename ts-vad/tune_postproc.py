"""Grid-search post-processing on VALIDATION probabilities (never on the eval/test set).

usage:
  python tune_postproc.py --probs exps/final/val/probs.npz --ref <dev>/val_all.rttm --out exps/final/postproc.json

probs.npz is written by `main.py --eval` (SAVE_PATH/probs.npz). The output JSON can be passed to
main.py with --postproc_json to apply the chosen settings to the eval set.
"""
import argparse, itertools, json, os, sys, tempfile
from multiprocessing import Pool
import postprocess, scoring_utils

def floats(s):
	return [float(x) for x in s.split(',') if x.strip()]

def ints(s):
	return [int(x) for x in s.split(',') if x.strip()]

_PROBS, _REF, _COLLAR, _TMP = None, None, None, None

def _init(probs_path, ref, collar, tmp):
	global _PROBS, _REF, _COLLAR, _TMP
	_PROBS, _REF, _COLLAR, _TMP = postprocess.load_probs(probs_path), ref, collar, tmp

def _score(params):
	threshold, median_kernel, min_silence, min_speech = params
	path = os.path.join(_TMP, 'sys_%d.rttm' % os.getpid())
	postprocess.write_rttm(_PROBS, path, threshold=threshold, median_kernel=median_kernel, min_silence=min_silence, min_speech=min_speech)
	r = scoring_utils.md_eval(_REF, path, _COLLAR)
	return dict(threshold=threshold, median_kernel=median_kernel, min_silence=min_silence, min_speech=min_speech, **r)

def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--probs', required=True, help='probs.npz from main.py --eval on the validation list')
	ap.add_argument('--ref', required=True, help='ground-truth RTTM of the same recordings (val_all.rttm)')
	ap.add_argument('--out', required=True, help='output JSON (best settings + full grid)')
	ap.add_argument('--collar', type=float, default=0.0, help='collar used for selection (DIHARD metric: 0)')
	ap.add_argument('--thresholds', default='0.3,0.35,0.4,0.45,0.5,0.55,0.6')
	ap.add_argument('--median_kernels', default='1,11,21,31')
	ap.add_argument('--min_silences', default='0,0.16,0.32,0.48')
	ap.add_argument('--min_speeches', default='0,0.08,0.16')
	ap.add_argument('--n_jobs', type=int, default=os.cpu_count() or 1)
	a = ap.parse_args()

	grid = list(itertools.product(floats(a.thresholds), ints(a.median_kernels), floats(a.min_silences), floats(a.min_speeches)))
	print('Scoring %d post-processing settings on %s with %d workers ...' % (len(grid), a.probs, a.n_jobs))
	with tempfile.TemporaryDirectory() as tmp:
		with Pool(a.n_jobs, initializer=_init, initargs=(a.probs, a.ref, a.collar, tmp)) as pool:
			results = []
			for i, r in enumerate(pool.imap_unordered(_score, grid), 1):
				results.append(r)
				sys.stderr.write('\r%d/%d' % (i, len(grid)))
	sys.stderr.write('\n')
	results.sort(key=lambda r: r['DER'])
	default = next((r for r in results if (r['threshold'], r['median_kernel'], r['min_silence'], r['min_speech']) == (0.5, 21, 0.32, 0.0)), None)

	print('\nTop 10 (collar %.2f):' % a.collar)
	print('%9s %6s %11s %10s %8s %7s %7s %7s' % ('threshold', 'median', 'min_silence', 'min_speech', 'DER', 'MS', 'FA', 'SC'))
	for r in results[:10]:
		print('%9.2f %6d %11.2f %10.2f %8.2f %7.2f %7.2f %7.2f' % (r['threshold'], r['median_kernel'], r['min_silence'], r['min_speech'], r['DER'], r['MS'], r['FA'], r['SC']))
	if default:
		print('\nDefault settings (0.5, 21, 0.32, 0): DER %.2f%%  ->  best: DER %.2f%%' % (default['DER'], results[0]['DER']))

	best = {k: results[0][k] for k in postprocess.DEFAULTS}
	json.dump({'best': best, 'selected_on': a.probs, 'reference': a.ref, 'collar': a.collar,
			   'best_scores': {k: results[0][k] for k in ('DER', 'MS', 'FA', 'SC')}, 'grid': results}, open(a.out, 'w'), indent=1)
	print('\nBest settings written to %s: %s' % (a.out, best))

if __name__ == '__main__':
	main()
