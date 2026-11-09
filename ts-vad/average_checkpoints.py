"""Average the weights of several checkpoints (usually more stable than any single epoch).

usage:
  python average_checkpoints.py --save_path exps/run1 --best 5 --out exps/run1/model/avg_best5.model
  python average_checkpoints.py --save_path exps/run1 --last 5 --out exps/run1/model/avg_last5.model

--best N picks the N epochs with the lowest collar-0 DER in SAVE_PATH/score.txt (validation runs).
Floating-point tensors are averaged; integer buffers (e.g. BatchNorm counters) come from the newest model.
"""
import argparse, glob, os, re
from collections import OrderedDict
import torch

def epoch_of(path):
	m = re.search(r'model_(\d+)\.model$', path)
	return int(m.group(1)) if m else None

def val_ders(score_file):
	"""{epoch: DER} from 'Eval full 0.00: <epoch> epoch, DER x%' lines (last value wins on reruns)."""
	ders = {}
	for line in open(score_file):
		m = re.match(r'Eval full 0\.00: (\d+) epoch, DER ([\d.]+)%', line)
		if m:
			ders[int(m.group(1))] = float(m.group(2))
	return ders

def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument('--save_path', required=True, help='experiment folder containing model/ and score.txt')
	g = ap.add_mutually_exclusive_group(required=True)
	g.add_argument('--best', type=int, help='average the N epochs with the lowest validation DER')
	g.add_argument('--last', type=int, help='average the N most recent epochs')
	ap.add_argument('--out', required=True)
	a = ap.parse_args()

	models = {epoch_of(p): p for p in glob.glob(os.path.join(a.save_path, 'model', 'model_*.model')) if epoch_of(p) is not None}
	if not models:
		raise SystemExit('No model_XXXX.model files in %s/model' % a.save_path)
	if a.best:
		ders = {e: d for e, d in val_ders(os.path.join(a.save_path, 'score.txt')).items() if e in models}
		if not ders:
			raise SystemExit('No validation DER lines for existing checkpoints in score.txt (train with --eval_epochs all)')
		epochs = sorted(ders, key=ders.get)[:a.best]
		print('Best %d epochs by validation DER: %s' % (len(epochs), ', '.join('%d (%.2f%%)' % (e, ders[e]) for e in sorted(epochs))))
	else:
		epochs = sorted(models)[-a.last:]
		print('Last %d epochs: %s' % (len(epochs), epochs))

	newest = torch.load(models[max(epochs)], map_location='cpu')
	avg = OrderedDict()
	for k, v in newest.items():
		avg[k] = v.clone().double() if torch.is_floating_point(v) else v.clone()
	for e in epochs:
		if e == max(epochs):
			continue
		sd = torch.load(models[e], map_location='cpu')
		for k, v in avg.items():
			if torch.is_floating_point(v):
				v += sd[k].double()
	for k, v in avg.items():
		if torch.is_floating_point(v):
			avg[k] = (v / len(epochs)).to(newest[k].dtype)
	torch.save(avg, a.out)
	print('Averaged %d checkpoints -> %s' % (len(epochs), a.out))

if __name__ == '__main__':
	main()
