"""Turn per-frame speaker probabilities into RTTM segments.

probs: {"<recording>-<speaker_id>": 1-D numpy array of P(speaking) per 40 ms frame}
params: threshold, median_kernel (odd, 1 = off), min_silence (s), min_speech (s)

Frame i covers [i*0.04, (i+1)*0.04). The original writer recorded each segment's start at the
index of the preceding silent frame, shifting every segment (except one starting at t=0) 40 ms
early; on the bundled res.rttm that cost 0.56 DER points at collar 0. Fixed here.
"""
import json, numpy
from scipy import signal

FRAME = 0.04
DEFAULTS = {'threshold': 0.5, 'median_kernel': 21, 'min_silence': 0.32, 'min_speech': 0.0}

def fill_short_silences(binary, min_silence):
	"""Silence runs of <= int(min_silence // 0.04) frames between (or at the edges of) speech become speech."""
	thr = int(min_silence // FRAME)
	out = binary.copy()
	n, i = len(binary), 0
	while i < n:
		if binary[i] == 0:
			j = i
			while j < n and binary[j] == 0:
				j += 1
			if j - i <= thr:
				out[i:j] = 1
			i = j
		else:
			i += 1
	return out

def drop_short_speech(binary, min_speech):
	"""Speech runs of <= int(min_speech // 0.04) frames become silence."""
	thr = int(min_speech // FRAME)
	out = binary.copy()
	n, i = len(binary), 0
	while i < n:
		if binary[i] == 1:
			j = i
			while j < n and binary[j] == 1:
				j += 1
			if j - i <= thr:
				out[i:j] = 0
			i = j
		else:
			i += 1
	return out

def binarize(prob, threshold=0.5, median_kernel=21, min_silence=0.32, min_speech=0.0):
	p = numpy.asarray(prob, dtype=numpy.float64)
	if median_kernel and median_kernel > 1 and len(p) > 0:
		k = int(median_kernel) | 1  # medfilt needs an odd kernel
		p = signal.medfilt(p, k)
	b = (p >= threshold).astype(numpy.int8)
	b = fill_short_silences(b, min_silence)
	b = drop_short_speech(b, min_speech)
	return b

def segments(binary):
	"""[(start_s, duration_s)] for each run of 1s."""
	segs, n, i = [], len(binary), 0
	while i < n:
		if binary[i] == 1:
			j = i
			while j < n and binary[j] == 1:
				j += 1
			segs.append((i * FRAME, (j - i) * FRAME))
			i = j
		else:
			i += 1
	return segs

def split_key(key):
	name, spk = key.rsplit('-', 1)
	return name, spk

def write_rttm(probs, path, **params):
	p = dict(DEFAULTS); p.update({k: v for k, v in params.items() if v is not None})
	with open(path, 'w') as f:
		for key in sorted(probs):
			name, spk = split_key(key)
			for start, dur in segments(binarize(probs[key], p['threshold'], p['median_kernel'], p['min_silence'], p['min_speech'])):
				f.write('SPEAKER %s 1 %.3f %.3f <NA> <NA> %s <NA> <NA>\n' % (name, start, dur, spk))

def load_probs(path):
	with numpy.load(path) as z:
		return {k: z[k] for k in z.files}

def load_params(path):
	"""Read a tune_postproc.py output (or any JSON with the DEFAULTS keys)."""
	d = json.load(open(path))
	d = d.get('best', d)
	return {k: d[k] for k in DEFAULTS if k in d}
