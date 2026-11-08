import argparse, os, sys
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--moj', required=True)
ap.add_argument('--skrypt', default='solve_fast')
ap.add_argument('--cache', default='cache_extra')
ap.add_argument('--wagi', default='cache_extra/unet.pt')
ap.add_argument('--width', type=int, default=48)
ap.add_argument('--n_real', type=int, default=5000)
ap.add_argument('--tag', default='moj')
a = ap.parse_args()

sys.path.insert(0, os.path.abspath(a.moj))
S = __import__(a.skrypt)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
cache = os.path.join(a.moj, a.cache)
f_tr = np.load(os.path.join(cache, 'f_train.npy'), mmap_mode='r')
f_te = np.load(os.path.join(cache, 'f_test.npy'), mmap_mode='r')

perm = np.random.default_rng(0).permutation(a.n_real)
va = np.sort(perm[:500])
trs = np.sort(perm[500:])[:400]
smp = np.asarray(f_tr[trs], np.float32)
mu = smp.mean((0, 2, 3), dtype=np.float64).reshape(1, -1, 1, 1).astype(np.float32)
sd = (smp.std((0, 2, 3), dtype=np.float64) + 1e-4).reshape(1, -1, 1, 1).astype(np.float32)
del smp

net = S.PatchUNet(cin=f_tr.shape[1], w=a.width).to(dev)
net.load_state_dict(torch.load(os.path.join(a.moj, a.wagi), map_location=dev, weights_only=True))
print(f'wczytano {a.wagi}, urzadzenie {dev}')

os.makedirs('cache_v2', exist_ok=True)
Pv, Gv = S.infer(net, dev, f_tr, va, mu, sd)
np.savez_compressed(f'cache_v2/{a.tag}_val.npz', P=Pv.astype(np.float16), G=Gv)
rows = S.read_train(os.path.join(a.moj, 'p1_student'))
iou = np.mean([S.punkty(rows[i]['box'], S.best_rect(p[1])[0])
               for i, p in zip(va, Pv) if rows[i]['box'] != S.NONE])
print(f'walidacja: IoU ramki z samego U-Neta na obrazkach z latka {iou:.3f}')

N = len(f_te)
Pt = np.zeros((N, 3, 64, 64), np.float16); Gt = np.zeros(N, np.float32)
for s in range(0, N, 500):
    idx = np.arange(s, min(s + 500, N))
    p, g = S.infer(net, dev, f_te, idx, mu, sd)
    Pt[idx] = p.astype(np.float16); Gt[idx] = g
    print(f'   test {idx[-1] + 1}/{N}', flush=True)
np.savez_compressed(f'cache_v2/{a.tag}_test.npz', P=Pt, G=Gt)
print(f'zapisano cache_v2/{a.tag}_val.npz i cache_v2/{a.tag}_test.npz')
