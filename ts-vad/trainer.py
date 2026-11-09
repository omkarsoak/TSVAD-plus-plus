import torch, sys, os, tqdm, numpy, soundfile, time, pickle, glob, random, scipy, signal, subprocess
from tools.tools import *
from loss import *
from model.ts_vad import TS_VAD
import postprocess, scoring_utils
# from model.ts_vad_light import TS_VAD
from collections import defaultdict, OrderedDict
from torch.cuda.amp import autocast,GradScaler
from scipy import signal

def init_trainer(args):
	s = trainer(args)
	args.epoch = 1
	# When training, an existing checkpoint in save_path wins over --init_model so that a
	# re-queued cluster job resumes instead of restarting fine-tuning from epoch 1.
	resume = len(args.modelfiles) >= 1 and (args.train or args.init_model == "")
	if args.init_model != "" and not resume:
		print("Model %s loaded from pretrain!"%args.init_model)
		s.load_parameters(args.init_model)
	elif resume:
		print("Model %s loaded from previous state!"%args.modelfiles[-1])
		args.epoch = int(os.path.splitext(os.path.basename(args.modelfiles[-1]))[0][6:]) + 1
		s.load_parameters(args.modelfiles[-1])
	return s

class trainer(torch.nn.Module):
	def __init__(self, args):
		super(trainer, self).__init__()
		# GPU if available (cluster), otherwise CPU (local smoke tests). Mixed precision only on GPU.
		self.device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.use_amp         = self.device.type == 'cuda'
		if not self.use_amp:
			print("CUDA not available: running on CPU without mixed precision (slow; use only for small tests)")
		self.ts_vad 		 = TS_VAD(args).to(self.device)
		self.ts_loss         = Loss(args.max_speaker, legacy_head = args.legacy_head).to(self.device)
		self.optim           = torch.optim.AdamW(self.parameters(), lr = args.lr)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = args.test_step, gamma = args.lr_decay)
		# print("Model para number = %.2f"%(sum(param.numel() for param in self.ts_vad.parameters()) / 1e6))

	def train_network(self, args):
		self.train()
		scaler = GradScaler(enabled = self.use_amp)
		self.scheduler.step(args.epoch - 1)
		index, nloss = 0, 0
		lr = self.optim.param_groups[0]['lr']
		time_start = time.time()

		for num, (rs, ts, labels) in enumerate(args.trainLoader, start = 1):
			self.zero_grad()
			labels  = torch.tensor(labels, dtype=torch.float32).to(self.device)
			with autocast(enabled = self.use_amp):
				rs_embeds  = self.ts_vad.rs_forward(rs.to(self.device))
				ts_embeds  = self.ts_vad.ts_forward(ts.to(self.device))
				outs       = self.ts_vad.cat_forward(rs_embeds, ts_embeds)
				loss, _    = self.ts_loss.forward(outs, labels)	
			scaler.scale(loss).backward()
			scaler.step(self.optim)
			scaler.update()
			index += len(labels)
			nloss += loss.detach().cpu().numpy()
			time_used = time.time() - time_start
			sys.stderr.write("Train: [%2d] %.2f%% (est %.1f mins) Lr: %6f, Loss: %.5f\r"%\
			(args.epoch, 100 * (num / args.trainLoader.__len__()), time_used * args.trainLoader.__len__() / num / 60, \
			lr, nloss/(num)))
			sys.stderr.flush()			
		sys.stdout.write("\n")

		args.score_file.write("Train: %d epoch, LR %f, LOSS %f\n"%(args.epoch, lr, nloss/num))
		args.score_file.flush()
		return
		
	def eval_network(self, args):
		"""Sliding-window inference -> averaged per-frame probabilities -> RTTM -> DER.

		Saves SAVE_PATH/probs.npz ({"<rec>-<spk>": P(speaking) per 40 ms frame}) so post-processing
		can be re-tuned (tune_postproc.py) without re-running the network.
		"""
		self.eval()
		index, nloss, num = 0, 0, 0
		time_start = time.time()
		acc = {}  # "<rec>-<spk>" -> [sum of probabilities, number of windows] per frame
		for num, (rs, ts, labels, filename, speaker_id, start) in enumerate(args.evalLoader, start = 1):
			labels  = torch.tensor(labels, dtype=torch.float32).to(self.device)
			with torch.no_grad():
				rs_embeds  = self.ts_vad.rs_forward(rs.to(self.device))
				ts_embeds  = self.ts_vad.ts_forward(ts.to(self.device))
				outs       = self.ts_vad.cat_forward(rs_embeds, ts_embeds)
				loss, outs   = self.ts_loss.forward(outs, labels)
			B, S, T = outs.shape
			ids, starts = speaker_id.numpy(), start.numpy()
			for b in range(B):
				t0 = int(starts[b])
				for i in range(S):
					sid = int(ids[b, i])
					if sid == 0: # filler slot (speaker from another recording), not part of the output
						continue
					key = '%s-%d' % (filename[b], sid)
					if key not in acc:
						acc[key] = [numpy.zeros(t0 + T), numpy.zeros(t0 + T)]
					sums, cnts = acc[key]
					if len(sums) < t0 + T: # grow geometrically
						new_len = max(t0 + T, 2 * len(sums))
						sums = numpy.concatenate([sums, numpy.zeros(new_len - len(sums))])
						cnts = numpy.concatenate([cnts, numpy.zeros(new_len - len(cnts))])
						acc[key] = [sums, cnts]
					sums[t0:t0 + T] += outs[b, i]
					cnts[t0:t0 + T] += 1
			index += B
			nloss += loss.detach().cpu().numpy()
			time_used = time.time() - time_start
			sys.stderr.write("Eval: [%2d] %.2f%% (est %.1f mins) Loss: %.5f\r"%\
			(args.epoch, 100 * (num / args.evalLoader.__len__()), time_used * args.evalLoader.__len__() / num / 60, \
			nloss/(num)))
			sys.stderr.flush()
		num = max(num, 1)

		probs = {}
		for key, (sums, cnts) in acc.items():
			covered = numpy.nonzero(cnts)[0]
			L = covered[-1] + 1 if len(covered) else 0
			probs[key] = (sums[:L] / numpy.maximum(cnts[:L], 1)).astype(numpy.float32)
		numpy.savez_compressed(args.probs_save_path, **probs)

		params = {'threshold': args.threshold, 'median_kernel': args.median_kernel,
				  'min_silence': args.min_silence, 'min_speech': args.min_speech}
		postprocess.write_rttm(probs, args.rttm_save_path, **params)
		print('\n' + args.rttm_save_path)
		print('Post-processing: %s' % params)

		if not os.path.isfile(args.ref_rttm):
			print("Reference RTTM %s not found, skipping DER scoring. Output RTTM: %s" % (args.ref_rttm, args.rttm_save_path))
			args.score_file.write("Eval: %d epoch, LOSS %f (no reference RTTM, DER not computed)\n"%(args.epoch, nloss/num))
			args.score_file.flush()
			return None

		results = {}
		for collar in (0.25, 0.0):
			r = scoring_utils.md_eval(args.ref_rttm, args.rttm_save_path, collar)
			results[collar] = r
			print("Collar %.2f: DER %2.2f%%, MS %2.2f%%, FA %2.2f%%, SC %2.2f%%"%(collar, r['DER'], r['MS'], r['FA'], r['SC']))
			args.score_file.write("Eval full %.2f: %d epoch, DER %2.2f%%, MS %2.2f%%, FA %2.2f%%, SC %2.2f%%, LOSS %f\n"%(collar, args.epoch, r['DER'], r['MS'], r['FA'], r['SC'], nloss/num))
		if args.uem_dir and os.path.isdir(args.uem_dir):
			table = scoring_utils.format_table(scoring_utils.der_by_domain(args.ref_rttm, args.rttm_save_path, args.uem_dir, 0.0))
			print("Per-domain DER (collar 0):\n" + table)
			args.score_file.write("Per-domain (collar 0), epoch %d:\n%s\n" % (args.epoch, table))
		args.score_file.flush()
		return results[0.0]['DER']

	def save_parameters(self, path):
		model = OrderedDict(list(self.ts_vad.state_dict().items()) + list(self.ts_loss.state_dict().items()))
		torch.save(model, path)

	def load_parameters(self, path):
		selfState = self.state_dict()
		loadedState = torch.load(path, map_location = self.device)
		for name, param in loadedState.items():
			origName = name
			if name not in selfState:
				name = 'ts_vad.' + name
				if name not in selfState:
					name = name.replace('ts_vad', 'ts_loss')
					if name not in selfState:
						print("%s is not in the model."%origName)
						continue
			if selfState[name].size() != loadedState[origName].size():
				sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s\n"%(origName, selfState[name].size(), loadedState[origName].size()))
				if 'fc' in origName:
					sys.stderr.write("  -> output head shape differs. If this checkpoint was trained before the head fix, "
						"evaluate it with --legacy_head true; when fine-tuning, the new head is trained from scratch.\n")
				continue
			selfState[name].copy_(param)
		