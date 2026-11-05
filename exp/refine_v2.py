"""Dopracowanie krawedzi ramki z iloczynu map (U-Net x przyczynowa) testem przyczynowym.
Dla kazdego wariantu krawedzi (+-OFF px) liczymy margines klasy modelu po zastapieniu ramki
wersja JPEG q30 i wybieramy regula strojona walidacja krzyzowa."""
import sys, itertools, time
import numpy as np
import torch
sys.path.insert(0, 'exp')
from common import *

OFF = (-4, -2, 0, 2, 4)


def variants(b):
    out = []
    for d in itertools.product(OFF, repeat=4):
        r = (max(b[0] + d[0], 0), max(b[1] + d[1], 0), min(b[2] + d[2], 64), min(b[3] + d[3], 64))
        if r[2] - r[0] >= 6 and r[3] - r[1] >= 6:
            out.append(r)
    return list(dict.fromkeys(out))


@torch.no_grad()
def rect_margins(bb, u8, rects, dev, q=30):
    x = to_t(u8[None]).to(dev); f = to_t(jpeg_u8(u8, q)[None]).to(dev)
    M = torch.zeros(len(rects), 1, 64, 64, device=dev)
    for j, r in enumerate(rects):
        M[j, :, r[1]:r[3], r[0]:r[2]] = 1
    with torch.autocast('cuda', dtype=torch.float16):
        z0 = bb(torch.cat([x, f])).float()
        c0 = z0[:1].argmax(1)
        Z = torch.cat([bb(x + M[s:s + 512] * (f - x)).float() for s in range(0, len(M), 512)])
    m0, mf = margin(z0, c0.repeat(2)).tolist()
    return (m0 - margin(Z, c0.repeat(len(Z))).cpu().numpy()) / (abs(m0 - mf) + 0.5)


def table(bb, U8, P1, C, dev):
    """-> lista (rects, cechy) na obrazek; cechy: drop, pole/100, oczekiwane IoU na mapie, start?"""
    T = []
    for u, p, c in zip(U8, P1, C):
        cz = np.clip(c.mean(0), 0, None); cz = cz / (cz.max() + 1e-9)
        pc = p * cz
        b0, _ = best_rect(pc)
        rects = variants(b0)
        I = integral(pc.astype(np.float64)); A = pc.sum()
        dr = rect_margins(bb, u, rects, dev)
        F = []
        for r, d in zip(rects, dr):
            a = (r[2] - r[0]) * (r[3] - r[1])
            s = I[r[3], r[2]] - I[r[1], r[2]] - I[r[3], r[0]] + I[r[1], r[0]]
            F.append([d, a / 100, s / (a + A - s + 1e-9), float(r == b0)])
        T.append((rects, np.array(F)))
    return T


def score(F, al, ga):
    return np.tanh(F[:, 0]) - al * F[:, 1] + ga * F[:, 2]


def choose(F, al, ga):
    return int(np.argmax(score(F, al, ga)))


def _iou_mat(rects):
    n = len(rects)
    M = np.zeros((n, n), np.float64)
    for i, a in enumerate(rects):
        aa = (a[2] - a[0]) * (a[3] - a[1])
        for j in range(i, n):
            b = rects[j]
            w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = w * h
            bb = (b[2] - b[0]) * (b[3] - b[1])
            M[i, j] = M[j, i] = inter / (aa + bb - inter) if inter else 0.0
    return M


def choose_mbr(rects, F, al, ga, temp):
    """Softmax po wyniku → wariant max. oczekiwane IoU względem rozkladu w."""
    s = score(F, al, ga)
    t = max(float(temp), 1e-6)
    w = np.exp((s - s.max()) / t); w /= w.sum()
    return int(np.argmax(_iou_mat(rects) @ w))


GRID = [(al, ga) for al in (0, .02, .05, .1, .2, .4) for ga in (0, 1, 2, 4, 8, 16, 1e6)]
TEMPS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)


def tune(T, gts):
    best = max(GRID, key=lambda p: np.mean([punkty(g, r[choose(F, *p)]) for (r, F), g in zip(T, gts)]))
    return best


def tune_mbr(T, gts, al, ga):
    """Strojenie temperatury MBR przy ustalonych (al, ga)."""
    return max(TEMPS, key=lambda t: np.mean(
        [punkty(g, r[choose_mbr(r, F, al, ga, t)]) for (r, F), g in zip(T, gts)]))


if __name__ == '__main__':
    dev = 'cuda'
    bb = backbone(dev)
    rows = train_rows()
    va = np.sort(np.random.default_rng(0).permutation(len(rows))[:500])
    gt = [rows[i]['box'] for i in va]; adv = np.array([g != NONE for g in gt])
    U8 = load_imgs([rows[i]['id'] for i in va[adv]], 'train')
    P1 = np.load('cache_v2/unet_val.npz')['P'][adv, 1].astype(np.float32)
    C = np.load('cache_v2/causal_val.npy')[adv].astype(np.float32)
    G = [g for g in gt if g != NONE]
    t = time.time(); T = table(bb, U8, P1, C, dev); print(f'{time.time() - t:.0f}s')
    st = np.mean([punkty(g, r[int(np.argmax(F[:, 3]))]) for (r, F), g in zip(T, G)])
    orc = np.mean([max(punkty(g, x) for x in r) for (r, F), g in zip(T, G)])
    fold = np.random.default_rng(0).integers(0, 5, len(T)); cv = np.zeros(len(T))
    for k in range(5):
        tr = np.nonzero(fold != k)[0]; te = np.nonzero(fold == k)[0]
        p = tune([T[i] for i in tr], [G[i] for i in tr])
        for i in te:
            cv[i] = punkty(G[i], T[i][0][choose(T[i][1], *p)])
    prm = tune(T, G)
    cv_mbr = np.zeros(len(T))
    for k in range(5):
        tr = np.nonzero(fold != k)[0]; te = np.nonzero(fold == k)[0]
        p = tune([T[i] for i in tr], [G[i] for i in tr])
        t_ = tune_mbr([T[i] for i in tr], [G[i] for i in tr], *p)
        for i in te:
            r, F = T[i]; cv_mbr[i] = punkty(G[i], r[choose_mbr(r, F, *p, t_)])
    print(f'start {st:.3f}  po dopracowaniu (5-fold CV) {cv.mean():.3f}  MBR {cv_mbr.mean():.3f}  '
          f'wyrocznia {orc:.3f}  parametry (calosc): {prm}  T_mbr={tune_mbr(T, G, *prm)}')
