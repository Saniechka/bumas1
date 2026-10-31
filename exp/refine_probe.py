"""Lokalne poprawianie ramki z U-Neta testem przyczynowym. Zapisuje tabele cech wariantow."""
import sys, itertools, time
import numpy as np
import torch
sys.path.insert(0, '.'); sys.path.insert(0, 'exp')
from common import *
import ensemble4 as E

dev = 'cuda'
bb = backbone(dev)
rows = train_rows()
name = sys.argv[1] if len(sys.argv) > 1 else 'oryginal'
Z = np.load(f'exp/val_{name}.npz')
P, G, U8, va = Z['P'].astype(np.float32), Z['G'], Z['u8'], Z['va']
OFF = (-6, -3, 0, 3, 6)
DEN = int(sys.argv[2]) if len(sys.argv) > 2 else 50


def rect_feats(x, f, rects, c0):
    M = torch.zeros(len(rects), 1, 64, 64)
    for j, r in enumerate(rects):
        M[j, :, r[1]:r[3], r[0]:r[2]] = 1
    out = []
    for s in range(0, len(rects), 256):
        m = M[s:s + 256].to(dev)
        X = torch.cat([x + m * (f - x), f + m * (x - f)])
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            z = bb(X).float()
        mg = margin(z, c0.repeat(len(z))).cpu().numpy()
        out.append(np.stack([mg[:len(m)], mg[len(m):]], 1))
    return np.concatenate(out)


tab = []
t = time.time()
for k, i in enumerate(va):
    gt = rows[i]['box']
    p = P[k]
    start = E.best_rect(p[1])[0]
    I = integral(p[1].astype(np.float64)); A = float(p[1].sum())
    rects = []
    for d in itertools.product(OFF, repeat=4):
        r = (max(start[0] + d[0], 0), max(start[1] + d[1], 0), min(start[2] + d[2], 64), min(start[3] + d[3], 64))
        if r[2] - r[0] >= 6 and r[3] - r[1] >= 6:
            rects.append(r)
    rects = list(dict.fromkeys(rects))
    x = to_t(U8[k:k + 1]).to(dev); f = to_t(jpeg_u8(U8[k], DEN)[None]).to(dev)
    with torch.no_grad():
        z0 = bb(torch.cat([x, f])).float()
    c0 = z0[0:1].argmax(1)
    m0 = margin(z0, c0.repeat(2)).cpu().numpy()
    mg = rect_feats(x, f, rects, c0)
    for r, (md, mp) in zip(rects, mg):
        a = (r[2] - r[0]) * (r[3] - r[1])
        s = I[r[3], r[2]] - I[r[1], r[2]] - I[r[3], r[0]] + I[r[1], r[0]]
        tab.append((k, a, md, mp, m0[0], m0[1], s / (a + A - s + 1e-9), s / a, float(r == start),
                    punkty(gt, r) if gt != NONE else -1, *r))
    if k % 100 == 0:
        print(k, f'{time.time() - t:.0f}s', flush=True)
np.save(f'exp/refine_{name}_q{DEN}.npy', np.array(tab, np.float32))
