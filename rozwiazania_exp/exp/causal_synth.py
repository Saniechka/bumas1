import sys, csv, re
import numpy as np
from PIL import Image
sys.path.insert(0, 'exp')
from common import *
from causal_map import causal_maps

dev = 'cuda'
bb = backbone(dev)
R = [r for r in csv.DictReader(open('synth_final/train_labels.csv')) if r['comment'].startswith('ADV')]
q = np.array([int(re.search(r'jpeg=(\d+)', r['comment']).group(1)) for r in R])
rng = np.random.default_rng(0)
sel = np.concatenate([rng.choice(np.nonzero(q < 89)[0], 200, replace=False),
                      rng.choice(np.nonzero(q >= 89)[0], 100, replace=False)])
U = np.stack([np.asarray(Image.open(f'synth_final/train/images/{R[i]["id"]}.png').convert('RGB')) for i in sel])
for dq in (50, 30):
    C, st = causal_maps(bb, U, dev, q=dq)
    for nm, m in (('q<89', q[sel] < 89), ('q>=89', q[sel] >= 89)):
        iou = []
        for c, i in zip(C[m], sel[m]):
            g = tuple(int(R[i][k]) for k in ('x_min', 'y_min', 'x_max', 'y_max'))
            c = np.clip(c[0], 0, None); iou.append(punkty(g, best_rect(c / (c.max() + 1e-9))[0]))
        iou = np.array(iou)
        print(f'odszumianie JPEG{dq}  {nm:6s} IoU usuwanie {iou.mean():.3f}  ~0 {(iou < .05).mean():.3f}', flush=True)
