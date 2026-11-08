"""Mapa przyczynowa: przesuwane okna, w oknie obraz zastapiony odszumionym (JPEG q).
Wartosc piksela = sredni spadek marginesu c0 po oknach, ktore go zawieraja."""
import sys, time
import numpy as np
import torch
sys.path.insert(0, 'exp')
from common import *

WINS = [(16, 16), (20, 20), (24, 24), (14, 28), (28, 14)]
STRIDE = 4


def win_masks(dev):
    ms = []
    for w, h in WINS:
        for y in list(range(0, 64 - h + 1, STRIDE)) + [64 - h]:
            for x in list(range(0, 64 - w + 1, STRIDE)) + [64 - w]:
                m = torch.zeros(64, 64, device=dev); m[y:y + h, x:x + w] = 1; ms.append(m)
    return torch.unique(torch.stack(ms), dim=0)


@torch.no_grad()
def causal_maps(bb, u8, dev, q=50, bs=512):
    """u8 (n,64,64,3) -> mapy (n,2,64,64): [spadek przy usuwaniu, wzrost przy zachowaniu], statystyki (n,4)."""
    M = win_masks(dev); cnt = M.sum(0)
    out = np.zeros((len(u8), 2, 64, 64), np.float32); st = np.zeros((len(u8), 4), np.float32)
    for n in range(len(u8)):
        x = to_t(u8[n:n + 1]).to(dev); f = to_t(jpeg_u8(u8[n], q)[None]).to(dev)
        with torch.autocast('cuda', dtype=torch.float16):
            z0 = bb(torch.cat([x, f])).float()
        c0 = z0[:1].argmax(1); m0, mf = margin(z0, c0.repeat(2)).tolist()
        md, mp = [], []
        for s in range(0, len(M), bs // 2):
            m = M[s:s + bs // 2, None]
            with torch.autocast('cuda', dtype=torch.float16):
                z = bb(torch.cat([x + m * (f - x), f + m * (x - f)])).float()
            mg = margin(z, c0.repeat(len(z)))
            md.append(mg[:len(m)]); mp.append(mg[len(m):])
        md, mp = torch.cat(md), torch.cat(mp)
        den = abs(m0 - mf) + 0.5
        drop = (m0 - md) / den; keep = (mp - mf) / den
        out[n, 0] = (torch.einsum('w,wij->ij', drop, M) / cnt).cpu().numpy()
        out[n, 1] = (torch.einsum('w,wij->ij', keep, M) / cnt).cpu().numpy()
        st[n] = [m0, mf, drop.max().item(), keep.max().item()]
    return out, st


if __name__ == '__main__':
    dev = 'cuda'
    bb = backbone(dev)
    rows = train_rows()
    name = sys.argv[1] if len(sys.argv) > 1 else 'oryginal'
    Z = np.load(f'exp/val_{name}.npz')
    P, U8, va = Z['P'].astype(np.float32), Z['u8'], Z['va']
    print('okien:', len(win_masks(dev)))
    t = time.time()
    C, st = causal_maps(bb, U8, dev)
    print(f'{time.time() - t:.0f}s')
    np.savez_compressed(f'exp/causal_{name}.npz', C=C.astype(np.float16), st=st)
    gt = [rows[i]['box'] for i in va]; has = np.array([g != NONE for g in gt])

    def ev(nm, maps):
        iou = []
        for m, g in zip(maps, gt):
            if g == NONE:
                continue
            m = np.clip(m, 0, None); iou.append(punkty(g, best_rect(m / (m.max() + 1e-9))[0]))
        iou = np.array(iou)
        print(f'  {nm:28s} IoU {iou.mean():.3f}  ~0: {(iou < .05).mean():.3f}')
    ev('U-Net', P[:, 1])
    ev('usuwanie', C[:, 0]); ev('zachowanie', C[:, 1]); ev('usuwanie+zachowanie', C[:, 0] + C[:, 1])
    for a in (0.5, 1, 2):
        cz = np.clip(C[:, 0] + C[:, 1], 0, None); cz = cz / (cz.max((1, 2), keepdims=True) + 1e-9)
        ev(f'U-Net * causal^{a}', P[:, 1] * cz ** a)
