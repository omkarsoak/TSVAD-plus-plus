"""DER scoring helpers built on the unmodified NIST md-eval.pl (scoring/utils/md-eval.pl).

    md_eval(ref, sys, collar)                 -> {'DER', 'MS', 'FA', 'SC', 'scored'}  (percent of scored speaker time)
    der_by_domain(ref, sys, uem_dir, collar)  -> {domain: {...}, 'ALL': {...}}

CLI:  python scoring_utils.py --ref all.rttm --sys res_rttm [--uem_dir .../uem_scoring/full] [--collar 0]
"""
import argparse, glob, os, re, subprocess, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MD_EVAL = os.path.join(HERE, '..', 'scoring', 'utils', 'md-eval.pl')

_PATTERNS = {
	'scored': r'SCORED SPEAKER TIME\s*=\s*([\d.]+)',
	'MS': r'MISSED SPEAKER TIME\s*=\s*([\d.]+)',
	'FA': r'FALARM SPEAKER TIME\s*=\s*([\d.]+)',
	'SC': r'SPEAKER ERROR TIME\s*=\s*([\d.]+)',
}

def md_eval(ref_rttm, sys_rttm, collar=0.0, uem=None):
	"""Run md-eval and return DER and its components in percent (computed from seconds for precision)."""
	cmd = ['perl', MD_EVAL, '-c', str(collar), '-r', ref_rttm, '-s', sys_rttm]
	if uem:
		cmd += ['-u', uem]
	out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True).stdout
	secs = {}
	for k, p in _PATTERNS.items():
		m = re.search(p, out)
		secs[k] = float(m.group(1)) if m else 0.0
	if secs['scored'] <= 0:
		return {'DER': float('nan'), 'MS': float('nan'), 'FA': float('nan'), 'SC': float('nan'), 'scored': 0.0}
	res = {k: 100.0 * secs[k] / secs['scored'] for k in ('MS', 'FA', 'SC')}
	res['DER'] = res['MS'] + res['FA'] + res['SC']
	res['scored'] = secs['scored']
	m = re.search(r'OVERALL SPEAKER DIARIZATION ERROR\s*=\s*([\d.]+)', out)
	if m and abs(float(m.group(1)) - res['DER']) > 0.01:
		raise RuntimeError('md-eval parse mismatch: overall %s vs computed %.2f' % (m.group(1), res['DER']))
	return res

def rttm_recordings(path):
	recs = set()
	for line in open(path):
		p = line.split()
		if len(p) >= 8 and p[0] == 'SPEAKER':
			recs.add(p[1])
	return recs

def filter_rttm(path, recs, out_path):
	with open(out_path, 'w') as f:
		for line in open(path):
			p = line.split()
			if len(p) >= 8 and p[0] == 'SPEAKER' and p[1] in recs:
				f.write(line)

def read_domains(uem_dir):
	"""DIHARD layout: uem_scoring/{full,core}/<domain>.uem with lines '<rec> <chan> <start> <end>'."""
	domains = {}
	for uem in sorted(glob.glob(os.path.join(uem_dir, '*.uem'))):
		name = os.path.splitext(os.path.basename(uem))[0]
		if name.lower() == 'all':
			continue
		recs = set(l.split()[0] for l in open(uem) if l.strip())
		domains[name] = (uem, recs)
	return domains

def der_by_domain(ref_rttm, sys_rttm, uem_dir, collar=0.0):
	"""Per-domain DER, restricted to recordings present in the reference (e.g. only the validation split)."""
	ref_recs = rttm_recordings(ref_rttm)
	results = {}
	with tempfile.TemporaryDirectory() as tmp:
		for name, (uem, recs) in read_domains(uem_dir).items():
			recs = recs & ref_recs
			if not recs:
				continue
			r, s = os.path.join(tmp, 'ref.rttm'), os.path.join(tmp, 'sys.rttm')
			filter_rttm(ref_rttm, recs, r)
			filter_rttm(sys_rttm, recs, s)
			results[name] = md_eval(r, s, collar, uem)
			results[name]['n'] = len(recs)
	results['ALL'] = md_eval(ref_rttm, sys_rttm, collar)
	results['ALL']['n'] = len(ref_recs)
	return results

def format_table(results):
	lines = ['%-22s %4s %8s %8s %8s %8s' % ('domain', 'n', 'DER', 'MS', 'FA', 'SC')]
	for name, r in results.items():
		lines.append('%-22s %4d %8.2f %8.2f %8.2f %8.2f' % (name, r.get('n', 0), r['DER'], r['MS'], r['FA'], r['SC']))
	return '\n'.join(lines)

if __name__ == '__main__':
	ap = argparse.ArgumentParser(description='DER (and optional per-domain DER) with md-eval')
	ap.add_argument('--ref', required=True)
	ap.add_argument('--sys', required=True)
	ap.add_argument('--uem_dir', default='', help='e.g. third_dihard_challenge_eval/data/uem_scoring/full')
	ap.add_argument('--collar', type=float, default=0.0)
	a = ap.parse_args()
	if a.uem_dir:
		print(format_table(der_by_domain(a.ref, a.sys, a.uem_dir, a.collar)))
	else:
		r = md_eval(a.ref, a.sys, a.collar)
		print('DER %.2f%%  MS %.2f%%  FA %.2f%%  SC %.2f%%' % (r['DER'], r['MS'], r['FA'], r['SC']))
