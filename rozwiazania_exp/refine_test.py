"""
refine_test.py - ile da sie zyskac na dokladniejszych krawedziach ramki?
Na walidacji (obrazki z latka) liczy dla roznych siatek przesuniec krawedzi:
  start     - IoU ramki z map (bez dopracowania)
  CV        - IoU po dopracowaniu regula z refine_v2 (5-fold, uczciwie)
  wyrocznia - IoU najlepszego wariantu (gorna granica, gdyby wybor byl idealny)

    python refine_test.py --unet unet,moj,unet3
"""
import argparse, sys, time
import numpy as np
import torch
sys.path.insert(0, '.'); sys.path.insert(0, 'exp')
import solve_v2 as S
import refine_v2 as R
from common import punkty, NONE, train_rows, load_imgs, backbone

ap = argparse.ArgumentParser()
ap.add_argument('--unet', default='unet,moj,unet3')
a = ap.parse_args()

rows = train_rows()
va = np.sort(np.random.default_rng(0).permutation(len(rows))[:500])
gt = [rows[i]['box'] for i in va]
adv = np.array([g != NONE for g in gt])
G = [g for g in gt if g != NONE]
P, _ = S.cached_unet('val', 'cpu', lambda: None, lambda: None, a.unet.split(','))
P1 = P[adv, 1].astype(np.float32)
C = np.load('cache_v2/causal_val.npy')[adv].astype(np.float32)
U8 = load_imgs([rows[i]['id'] for i in va[adv]], 'train')
dev = 'cuda'
bb = backbone(dev)

SIATKI = {
    'obecna  -4..4 co 2': (-4, -2, 0, 2, 4),
    'szersza -6..6 co 2': (-6, -4, -2, 0, 2, 4, 6),
    'gesta   -3..3 co 1': (-3, -2, -1, 0, 1, 2, 3),
}
for nazwa, off in SIATKI.items():
    R.OFF = off
    t = time.time()
    T = R.table(bb, U8, P1, C, dev)
    st = np.mean([punkty(g, r[int(np.argmax(F[:, 3]))]) for (r, F), g in zip(T, G)])
    orc = np.mean([max(punkty(g, x) for x in r) for (r, F), g in zip(T, G)])
    fold = np.random.default_rng(0).integers(0, 5, len(T)); cv = np.zeros(len(T))
    for k in range(5):
        tr = np.nonzero(fold != k)[0]; te = np.nonzero(fold == k)[0]
        p = R.tune([T[i] for i in tr], [G[i] for i in tr])
        for i in te:
            cv[i] = punkty(G[i], T[i][0][R.choose(T[i][1], *p)])
    n_war = np.mean([len(r) for r, _ in T])
    print(f'{nazwa}: start {st:.3f}  CV po dopracowaniu {cv.mean():.3f}  wyrocznia {orc:.3f}'
          f'   ({n_war:.0f} wariantow/obrazek, {time.time() - t:.0f}s)', flush=True)
