"""Czulosc testu przyczynowego na ramkach wokol prawdziwej latki.
pres(R): f poza R, x w R  -> czy model dalej daje c0 (zla klase)?
del(R):  x poza R, f w R  -> czy model zmienia c0?"""
import sys
import numpy as np
import torch
sys.path.insert(0, 'exp')
from common import *

dev = 'cuda'
bb = backbone(dev)
rows = train_rows()
rng = np.random.default_rng(1)
adv = [i for i, r in enumerate(rows) if r['box'] != NONE]
sel = np.array(sorted(rng.choice(adv, 600, replace=False)))
X = load_imgs([rows[i]['id'] for i in sel], 'train')


def variants(b):
    x0, y0, x1, y1 = b
    out = {}
    for d in (-6, -4, -2, 0, 2, 4, 6):
        out[f'skala {d:+d}'] = (max(x0 - d, 0), max(y0 - d, 0), min(x1 + d, 64), min(y1 + d, 64))
    out['przes. 4px'] = (x0 + 4, y0, min(x1 + 4, 64), y1) if x1 + 4 <= 64 else (x0 - 4, y0, x1 - 4, y1)
    out['przes. 8px'] = (x0 + 8, y0, min(x1 + 8, 64), y1) if x1 + 8 <= 64 else (x0 - 8, y0, x1 - 8, y1)
    out['lustro'] = (x0, 64 - y1, x1, 64 - y0)
    return out


for fname, fq in (('jpeg50', 50), ('jpeg30', 30)):
    Fq = np.stack([jpeg_u8(u, fq) for u in X])
    res = {}
    for k, i in enumerate(sel):
        b = rows[i]['box']
        x = to_t(X[k:k + 1]); f = to_t(Fq[k:k + 1])
        V = variants(b); keys = list(V)
        M = torch.zeros(len(keys), 1, 64, 64)
        for j, kk in enumerate(keys):
            r = V[kk]; M[j, :, r[1]:r[3], r[0]:r[2]] = 1
        pres = f + M * (x - f); dele = x + M * (f - x)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            z = bb(torch.cat([x, f, pres, dele]).to(dev)).float()
        c0 = z[0].argmax()
        mg = margin(z, c0.repeat(len(z))).cpu().numpy()
        n = len(keys)
        for j, kk in enumerate(keys):
            res.setdefault(kk, []).append((mg[2 + j] > 0, mg[2 + n + j] < 0, mg[2 + j], mg[2 + n + j],
                                           punkty(b, V[kk])))
        res.setdefault('_f', []).append(mg[1] < 0)
    print(f'== {fname}: odszumienie calosci zmienia klase w {np.mean(res["_f"]):.3f}')
    for kk, v in res.items():
        if kk == '_f':
            continue
        v = np.array(v, dtype=np.float64)
        print(f'  {kk:12s} IoU {v[:, 4].mean():.2f}  pres utrzymuje c0 {v[:, 0].mean():.3f}  '
              f'del zmienia c0 {v[:, 1].mean():.3f}   med. margines pres {np.median(v[:, 2]):+.2f} del {np.median(v[:, 3]):+.2f}')
