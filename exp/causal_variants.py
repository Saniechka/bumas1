"""Warianty mapy przyczynowej na walidacji: sposob odszumiania okna x miara x siatka okien.
Kryterium: IoU ramki z iloczynu P(latka) U-Neta i mapy przyczynowej."""
import sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, 'exp')
from common import *
import causal_map as CM

dev = 'cuda'
bb = backbone(dev)
rows = train_rows()
va = np.sort(np.random.default_rng(0).permutation(len(rows))[:500])
gt = [rows[i]['box'] for i in va]
adv = np.array([g != NONE for g in gt])
U8 = load_imgs([rows[i]['id'] for i in va[adv]], 'train')
GT = [g for g in gt if g != NONE]
P1 = np.load('cache_v2/unet_val.npz')['P'][adv, 1].astype(np.float32)


def med3(t):
    p = F.pad(t, (1, 1, 1, 1), mode='reflect')
    return p.unfold(2, 3, 1).unfold(3, 3, 1).reshape(*t.shape, 9).median(-1).values


def blur(t, s=1.0):
    k = torch.arange(-2, 3, dtype=t.dtype, device=t.device); g = torch.exp(-k ** 2 / (2 * s * s)); g /= g.sum()
    t = F.conv2d(F.pad(t, (2, 2, 0, 0), mode='reflect'), g.view(1, 1, 1, 5).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(t, (0, 0, 2, 2), mode='reflect'), g.view(1, 1, 5, 1).repeat(3, 1, 1, 1), groups=3)


DEN = {
    'jpeg30': lambda u, x: to_t(jpeg_u8(u, 30)[None]).to(dev),
    'jpeg50': lambda u, x: to_t(jpeg_u8(u, 50)[None]).to(dev),
    'med3': lambda u, x: med3(x),
    'blur1': lambda u, x: blur(x, 1.0),
}


@torch.no_grad()
def maps(den, M):
    cnt = M.sum(0)
    out = {k: np.zeros((len(U8), 64, 64), np.float32) for k in ('drop', 'restore', 'flip')}
    for n in range(len(U8)):
        x = to_t(U8[n:n + 1]).to(dev); f = DEN[den](U8[n], x)
        with torch.autocast('cuda', dtype=torch.float16):
            z0 = bb(torch.cat([x, f])).float()
        c0 = z0[0].argmax(); cf = z0[1].argmax()
        m0 = margin(z0[:1], c0[None]).item(); mf = margin(z0[1:], c0[None]).item()
        Z = []
        for s in range(0, len(M), 512):
            m = M[s:s + 512, None]
            with torch.autocast('cuda', dtype=torch.float16):
                Z.append(bb(x + m * (f - x)).float())
        Z = torch.cat(Z)
        md = margin(Z, c0.repeat(len(Z)))
        sc = dict(drop=(m0 - md) / (abs(m0 - mf) + 0.5),
                  restore=torch.sigmoid(Z[:, cf] - Z[:, c0]) if cf != c0 else torch.sigmoid(-md),
                  flip=(Z.argmax(1) != c0).float())
        for k, v in sc.items():
            out[k][n] = (torch.einsum('w,wij->ij', v, M) / cnt).cpu().numpy()
    return out


def score(C, a=1.0):
    r = []
    for c, p, g in zip(C, P1, GT):
        cz = np.clip(c, 0, None); cz = cz / (cz.max() + 1e-9)
        r.append(punkty(g, best_rect(p * cz ** a)[0]))
    return np.mean(r)


res = {}
for grid in sys.argv[1].split(','):
    if grid == 's4':
        CM.STRIDE = 4; CM.WINS = [(16, 16), (20, 20), (24, 24), (14, 28), (28, 14)]
    elif grid == 's2':
        CM.STRIDE = 2; CM.WINS = [(16, 16), (20, 20), (24, 24), (14, 28), (28, 14)]
    elif grid == 's4small':
        CM.STRIDE = 4; CM.WINS = [(8, 8), (12, 12), (16, 16), (10, 20), (20, 10)]
    M = CM.win_masks(dev)
    for den in sys.argv[2].split(','):
        t = time.time()
        mp = maps(den, M)
        for k, v in mp.items():
            res[(grid, den, k)] = v
            print(f'{grid:8s} {den:7s} {k:8s} okien {len(M):4d}  IoU {score(v):.3f}  (a=0.5: {score(v, .5):.3f})  {time.time() - t:.0f}s', flush=True)
    # kombinacje miar i odszumiaczy w tej siatce
    for k in ('drop', 'restore'):
        ks = [res[(grid, d, k)] for d in sys.argv[2].split(',')]
        norm = [np.clip(c, 0, None) / (np.clip(c, 0, None).max((1, 2), keepdims=True) + 1e-9) for c in ks]
        print(f'{grid:8s} srednia wszystkich odszumiaczy, {k}: IoU {score(np.mean(norm, 0)):.3f}', flush=True)
np.savez_compressed('exp/causal_variants.npz', **{'|'.join(k): v.astype(np.float16) for k, v in res.items()})
