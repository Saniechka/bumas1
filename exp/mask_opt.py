"""Lokalizacja latki optymalizacja maski miedzy obrazem x a jego odszumiona wersja f.
del:  x + m (f - x)  -> minimalna maska, ktora zmienia klase c0
pres: f + m (x - f)  -> minimalna maska szumu, ktora utrzymuje klase c0
"""
import sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, 'exp')
from common import *

dev = 'cuda'
bb = backbone(dev)
rows = train_rows()
rng = np.random.default_rng(0)
adv = [i for i, r in enumerate(rows) if r['box'] != NONE]
sel = np.array(sorted(rng.choice(adv, 384, replace=False)))
X = load_imgs([rows[i]['id'] for i in sel], 'train')
Fq = np.stack([jpeg_u8(u, 50) for u in X])


def opt_mask(x, f, mode, lam, steps=150, kappa=1.0, res=32, lr=0.15):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        c0 = bb(x).argmax(1)
    th = torch.full((len(x), 1, res, res), -2.0 if mode == 'del' else -1.0, device=dev, requires_grad=True)
    opt = torch.optim.Adam([th], lr=lr)
    for _ in range(steps):
        m = F.interpolate(torch.sigmoid(th), size=64, mode='bilinear', align_corners=False)
        xi = x + m * (f - x) if mode == 'del' else f + m * (x - f)
        with torch.autocast('cuda', dtype=torch.float16):
            z = bb(xi).float()
        mg = margin(z, c0)
        l = (mg + kappa).clamp_min(0) if mode == 'del' else (kappa - mg).clamp_min(0)
        loss = (l + lam * m.mean((1, 2, 3))).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        m = F.interpolate(torch.sigmoid(th), size=64, mode='bilinear', align_corners=False)
    return m[:, 0].cpu().numpy(), mg.detach().cpu().numpy()


def grad_map(x, f):
    x = x.clone().requires_grad_(True)
    z = bb(x); c0 = z.argmax(1)
    g = torch.autograd.grad(margin(z, c0).sum(), x)[0]
    return (-(g * (f - x.detach())).sum(1)).clamp_min(0).cpu().numpy()


def evaluate(name, maps):
    ious = []
    for k, i in enumerate(sel):
        m = maps[k].astype(np.float64)
        b, _ = best_rect(m / (m.max() + 1e-12))
        ious.append(punkty(rows[i]['box'], b))
    ious = np.array(ious)
    print(f'{name:22s} IoU {ious.mean():.3f}   >0.5: {(ious > 0.5).mean():.3f}   ~0: {(ious < 0.05).mean():.3f}', flush=True)
    return ious


x_all, f_all = to_t(X), to_t(Fq)
B = 96
t = time.time()
gm = np.concatenate([grad_map(x_all[s:s + B].to(dev), f_all[s:s + B].to(dev)) for s in range(0, len(sel), B)])
evaluate('grad x (f-x)', gm)
for mode in ('del', 'pres'):
    for lam in (2.0, 5.0, 15.0):
        t = time.time()
        ms = np.concatenate([opt_mask(x_all[s:s + B].to(dev), f_all[s:s + B].to(dev), mode, lam)[0]
                             for s in range(0, len(sel), B)])
        evaluate(f'{mode} lam={lam}', ms)
        print(f'   {time.time() - t:.0f}s, srednie pole maski {ms.mean():.3f}', flush=True)
