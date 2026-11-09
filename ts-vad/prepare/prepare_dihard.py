import glob, tqdm, os, soundfile, copy, json, argparse, numpy, torch, wave, heapq, shutil
from collections import defaultdict
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Segment(object):
	def __init__(self, uttid, spkr, stime, etime, text, name):
		self.uttid = uttid
		self.spkr = spkr
		self.stime = round(stime, 2)
		self.etime = round(etime, 2)
		self.text = text
		self.name = name

	def change_stime(self, time):
		self.stime = time

	def change_etime(self, time):
		self.etime = time

def remove_overlap(aa, bb):
	"""
	Remove overlapping intervals between two lists of intervals.

	Args:
		aa (list): The first list of intervals.
		bb (list): The second list of intervals.

	Returns:
		list: A new list of intervals with overlapping intervals removed.
	"""
	# Sort the intervals in both lists based on their start time
	a = aa.copy()
	b = bb.copy()
	a.sort()
	b.sort()

	# Initialize the new list of intervals
	result = []

	# Initialize variables to keep track of the current interval in list a and the remaining intervals in list b
	i = 0
	j = 0

	# Iterate through the intervals in list a
	while i < len(a):
		# If there are no more intervals in list b or the current interval in list a does not overlap with the current interval in list b, add it to the result and move on to the next interval in list a
		if j == len(b) or a[i][1] <= b[j][0]:
			result.append(a[i])
			i += 1
		# If the current interval in list a completely overlaps with the current interval in list b, skip it and move on to the next interval in list a
		elif a[i][0] >= b[j][0] and a[i][1] <= b[j][1]:
			i += 1
		# If the current interval in list a partially overlaps with the current interval in list b, add the non-overlapping part to the result and move on to the next interval in list a
		elif a[i][0] < b[j][1] and a[i][1] > b[j][0]:
			if a[i][0] < b[j][0]:
				result.append([a[i][0], b[j][0]])
			a[i][0] = b[j][1]
		# If the current interval in list a starts after the current interval in list b, move on to the next interval in list b
		elif a[i][0] >= b[j][1]:
			j += 1

	# Return the new list of intervals
	return result

def merge_intervals(ints):
	"""Merge overlapping [start, end] intervals of one speaker."""
	out = []
	for s, e in sorted(ints):
		if out and s <= out[-1][1]:
			out[-1][1] = max(out[-1][1], e)
		else:
			out.append([s, e])
	return out

def init_speaker_encoder(source):
	"""
	Initializes a speaker encoder model with pre-trained weights from a given source file.

	Args:
		source (str): The path to the source file containing the pre-trained weights.

	Returns:
		speaker_encoder (ECAPA_TDNN): The initialized speaker encoder model with pre-trained weights.
	"""
	pretrained_state = torch.load(source, map_location='cpu')
	enc_name = os.path.basename(source)[0:4].lower()
	if enc_name == 'ecap':
		# ECAPA-TDNN (default in this repo): pretrained_models/ecapa-tdnn.model
		from speaker_encoder import ECAPA_TDNN
		speaker_encoder = ECAPA_TDNN(C=1024)
		selfState = speaker_encoder.state_dict()
		loaded = 0
		for name, param in pretrained_state.items():
			# checkpoints saved from a training wrapper prefix keys with 'speaker_encoder.'
			key = name[len('speaker_encoder.'):] if name.startswith('speaker_encoder.') else name
			if key in selfState and selfState[key].size() == param.size():
				selfState[key].copy_(param)
				loaded += 1
		if loaded == 0:
			raise RuntimeError('No ECAPA-TDNN weights matched in %s' % source)
		print('Loaded %d/%d ECAPA-TDNN tensors from %s' % (loaded, len(selfState), source))
	elif enc_name == 'camp':
		from Cam_plusplus.DTDNN import CAMPPlus
		speaker_encoder = CAMPPlus(feat_dim=80, embedding_size=192)
		speaker_encoder.load_state_dict(pretrained_state, strict = False)
	elif enc_name == 'eres':
		from eres2net.ERes2Net import ERes2Net
		speaker_encoder = ERes2Net(feat_dim=80, embedding_size=192, baseWidth=24, scale=4, expansion=4)
		speaker_encoder.load_state_dict(pretrained_state, strict = False)
	else:
		raise ValueError("Unknown speaker encoder '%s'. File name must start with 'ecapa', 'campplus' or 'eres2net'." % source)
	speaker_encoder.to(DEVICE)
	speaker_encoder.eval()
	return speaker_encoder

def get_args():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--data_path', help='the path for dihard3')
	parser.add_argument('--max_speaker', type=int, help='max number of speakers')
	parser.add_argument('--type', help='dev or eval')
	parser.add_argument('--source', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pretrained_models', 'ecapa-tdnn.model'), help='path of the speaker encoder checkpoint (ecapa-tdnn.model, campplus*.pt or eres2net*.pt)')
	parser.add_argument('--length_embedding', type=float, default=6, help='length of embeddings, seconds')
	parser.add_argument('--step_embedding', type=float, default=1, help='step of embeddings, seconds')
	parser.add_argument('--batch_size', type=int, default=96, help='step of embeddings, seconds')
	parser.add_argument('--min_clean_dur', type=float, default=2.0, help='seconds of non-overlapped speech needed to build a profile from clean speech only')
	parser.add_argument('--min_speaker_dur', type=float, default=0.5, help='speakers with less profile audio than this are dropped')
	parser.add_argument('--overlap_fallback', type=lambda v: str(v).lower() in ('1', 'true', 'yes'), default=True,
		help='use overlapped speech for speakers below --min_clean_dur (false + --min_clean_dur 6 ~ original behaviour)')

	args = parser.parse_args()
	args.path = os.path.join(args.data_path, 'third_dihard_challenge_%s/data'%(args.type))
	# args.path_flac = os.path.join(args.path, 'flac')
	args.path_wav = os.path.join(args.path, 'wav')
	args.path_rttm = os.path.join(args.path, 'rttm')
	# args.target_flac = os.path.join(args.path, 'target_audio')
	args.target_wav = os.path.join(args.path, 'target_audio')
	args.target_embedding = os.path.join(args.path, 'target_embedding')
	args.out_text = os.path.join(args.path, 'ts_%s.json'%(args.type))
	return args

def save_target_audio(target_audio_path, segments, flac_path):
	# each segment is a Segment object with uttid, spkr, stime, etime, text, name

	# read the original audio
	audio, sr = soundfile.read(flac_path)

	# save the original audio
	soundfile.write(os.path.join(target_audio_path, 'all.flac'), audio, sr)

	# save the audio for each speaker using the segments. Segments have been sorted by stime. For each spkr, we combine all the segments and save as one flac file.

	# get the unique speakers
	speakers = set([seg.spkr for seg in segments])

	# for each speaker, combine the segments and save as one flac file
	for spkr in speakers:
		# get the segments for this speaker
		spkr_segments = [seg for seg in segments if seg.spkr == spkr]
		# combine the segments
		spkr_audio = numpy.concatenate([audio[int(seg.stime*sr):int(seg.etime*sr)] for seg in spkr_segments])
		# save the audio
		soundfile.write(os.path.join(target_audio_path, spkr + '.flac'), spkr_audio, sr)


def main():
	args = get_args()
	
	outs = open(args.out_text, "w")
	stats = defaultdict(float)
	# all.rttm is the concatenated ground truth used only for scoring; never treat it as a recording
	rttms = sorted(r for r in glob.glob(args.path_rttm + '/*.rttm') if os.path.basename(r) != 'all.rttm')

	for rttm in tqdm.tqdm(rttms):
		# convert rttm file to a json file.
		# The format of the rttm file is: SPEAKER <filename> 1 <turn_onset> <turn_duration> <NA> <NA> <speaker_id> <NA> <NA>
		# The format of the json file is: [ {"filename":"filename", "speaker_key":1, "speaker_id":"speakerid", "labels": [0,1,0]}]
		segments = []
		uttid2spkr = {}
		uttid2name = {}

		with open(rttm, 'r') as f:
			for line in f:
				line = line.strip().split()
				if len(line) < 8 or line[0] != 'SPEAKER':
					continue
				uttid = line[1]
				stime = float(line[3])
				etime = float(line[3]) + float(line[4])
				name = line[7]
				# spkr is line[7] without the prefix "speaker" (if present, e.g. DIHARD ground truth);
				# clustering outputs such as VBx often use bare labels like "1" or "spk01"
				spkr = line[7][7:] if line[7].startswith('speaker') else line[7]
				segments.append(Segment(uttid, spkr, stime, etime, '', name))
				uttid2spkr[uttid] = spkr
				uttid2name[uttid] = name

		segments = sorted(segments, key=lambda x: x.spkr)

		intervals = defaultdict(list)
		new_intervals = defaultdict(list)
		
		dic = defaultdict()
		# Summary the intervals for all speakers
		for i in range(len(segments)):
			interval = [segments[i].stime, segments[i].etime]
			intervals[segments[i].spkr].append(interval)
			dic[str(segments[i].uttid) + '_' + str(segments[i].spkr)] = segments[i].name.split('_')[-1]

		# Remove the overlapped speeech
		for key in intervals:
			new_interval = intervals[key]
			for o_key in intervals:
				if o_key != key:
					new_interval = remove_overlap(copy.deepcopy(new_interval), copy.deepcopy(intervals[o_key]))
			new_intervals[key] = new_interval
		wav_file = glob.glob(os.path.join(args.path_wav, uttid) + '*.wav')[0]
		orig_audio, _ = soundfile.read(wav_file)
		length = len(orig_audio) 

		id_full = os.path.basename(wav_file)[:-4]
		output_dir = os.path.join(args.target_wav, id_full)
		# start clean so stale speaker files from an earlier run cannot leak into this one
		shutil.rmtree(output_dir, ignore_errors = True)
		shutil.rmtree(os.path.join(args.target_embedding, id_full), ignore_errors = True)
		os.makedirs(output_dir, exist_ok = True)
		n_frames = int(length / 16000 * 25) # 40ms, one label

		def cut(ints):
			parts = [orig_audio[int(s * 16000):int(e * 16000)] for s, e in ints]
			return numpy.concatenate(parts) if parts else numpy.zeros(0)

		# Profile audio for each speaker: non-overlapped speech if there is enough of it, otherwise all of
		# the speaker's speech (overlap included). The original script silently dropped every speaker with
		# < 6 s of non-overlapped speech, which made all of their speech unrecoverable missed speech.
		candidates = {}
		for key in intervals:
			total = sum(max(0.0, e - s) for s, e in merge_intervals(intervals[key]))
			clean = sum(max(0.0, e - s) for s, e in new_intervals[key])
			if clean >= args.min_clean_dur:
				audio, source = cut(new_intervals[key]), 'clean'
			elif args.overlap_fallback:
				audio, source = cut(merge_intervals(intervals[key])), 'with_overlap'
			else:
				stats['dropped_short'] += 1; stats['dropped_speech_s'] += total
				continue
			if len(audio) < args.min_speaker_dur * 16000:
				stats['dropped_short'] += 1; stats['dropped_speech_s'] += total
				continue
			candidates[key] = {'total': total, 'audio': audio, 'source': source}

		# keep the max_speaker speakers who talk the most; ids are 1..N without gaps
		kept = sorted(candidates, key = lambda k: candidates[k]['total'], reverse = True)
		for key in kept[args.max_speaker:]:
			stats['dropped_max_speaker'] += 1; stats['dropped_speech_s'] += candidates[key]['total']
		kept = kept[:args.max_speaker]
		for idx, key in enumerate(kept, start = 1):
			soundfile.write(os.path.join(output_dir, '%d.wav' % idx), candidates[key]['audio'], 16000)
			stats['kept_' + candidates[key]['source']] += 1
			labels = [0] * n_frames
			for s, e in intervals[key]:
				for i in range(int(s * 25), min(int(e * 25) + 1, len(labels))):
					labels[i] = 1
			res = {'filename': id_full, 'speaker_key': str(idx), 'speaker_id': 'speaker' + str(key), 'labels': labels}
			json.dump(res, outs)
			outs.write('\n')
		soundfile.write(os.path.join(output_dir, 'all.wav'), orig_audio, 16000)

	outs.close()
	print('Speakers kept: %d with clean profile, %d with overlap-included profile (< %.1f s clean speech); '
		  'dropped: %d too short (< %.1f s), %d beyond max_speaker=%d; speech of dropped speakers: %.1f s'
		  % (stats['kept_clean'], stats['kept_with_overlap'], args.min_clean_dur, stats['dropped_short'], args.min_speaker_dur,
			 stats['dropped_max_speaker'], args.max_speaker, stats['dropped_speech_s']))

	# Extract embeddings
	files = sorted(glob.glob(args.target_wav + "/*/*.wav"))
	model = init_speaker_encoder(args.source)

	for file in tqdm.tqdm(files):
		if 'all.wav' not in file:
			embeddings = []
			wav_length = wave.open(file, 'rb').getnframes() # entire length for target speech
			win = int(args.length_embedding * 16000)
			starts = list(range(0, wav_length - win, int(args.step_embedding * 16000))) if wav_length > win else [0]
			for start in starts:
				stop = min(start + win, wav_length)
				target_speech, _ = soundfile.read(file, start = start, stop = stop)
				target_speech = torch.FloatTensor(numpy.array(target_speech))
				with torch.no_grad():
					embeddings.extend(model.forward(target_speech.unsqueeze(0).to(DEVICE)))
			if embeddings:
				embeddings = torch.stack(embeddings)
				output_file = args.target_embedding + '/' + file.split('/')[-2] + '/' + file.split('/')[-1].replace('.wav', '.pt')
				os.makedirs(os.path.dirname(output_file), exist_ok = True)
				torch.save(embeddings, output_file)
			else:
				print(file)
if __name__ == '__main__':
	main()
