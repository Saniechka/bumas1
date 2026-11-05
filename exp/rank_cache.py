"""
Etap 4: cechy wariantow ramki (cache wznawialny, batche fp16).

    python exp/rank_cache.py --split val
    python exp/rank_cache.py --split train
    python exp/rank_cache.py --split test
"""
import argparse, os, sys, time, itertools
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import median_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
import ensemble4 as E

OFF4 = (-4, -2, 0, 2, 4)
OFF8 = (-8, -4, 0, 4, 8)


def variants(b, offs=OFF4):
    out = []
    for d in itertools.product(offs, repeat=4):
        r = (max(b[0] + d[0], 0), max(b[1] + d[1], 0),
             min(b[2] + d[2], 64), min(b[3] + d[3], 64))
        if r[2] - r[0] >= 6 and r[3] - r[1] >= 6:
            out.append(r)
    return list(dict.fromkeys(out))


def top_starts(pc, k=2):
    cands = [best_rect(pc)[0]]
    for thr in (0.45, 0.65):
        m = pc >= thr * (pc.max() + 1e-9)
        if m.sum() < 16:
            continue
        ys, xs = np.where(m)
        b = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        if b[2] - b[0] >= 6 and b[3] - b[1] >= 6:
            cands.append(b)
    return list(dict.fromkeys(cands))[:k]


def gather_rects(starts, with8=False):
    seen, R, S0 = set(), [], []
    # starty NA POCZATKU (F[0] = glowny start)
    for s0 in starts:
        if s0 not in seen:
            seen.add(s0); R.append(s0); S0.append(s0)
    for s0 in starts:
        for r in variants(s0, OFF4):
            if r not in seen:
                seen.add(r); R.append(r); S0.append(s0)
        if with8:
            for r in variants(s0, OFF8):
                if r not in seen:
                    seen.add(r); R.append(r); S0.append(s0)
    return R, S0


@torch.no_grad()
def margins_many(bb, x, f, rects, dev, mode='remove', bs=512):
    if not rects:
        return np.zeros(0, np.float32)
    M = torch.zeros(len(rects), 1, 64, 64, device=dev)
    for j, r in enumerate(rects):
        M[j, :, r[1]:r[3], r[0]:r[2]] = 1
    with torch.autocast('cuda', dtype=torch.float16):
        z0 = bb(torch.cat([x, f])).float()
        c0 = z0[:1].argmax(1)
        m0, mf = margin(z0, c0.repeat(2))
        outs = []
        for s in range(0, len(M), bs):
            m = M[s:s + bs]
            xx = x + m * (f - x) if mode == 'remove' else f + m * (x - f)
            outs.append(margin(bb(xx).float(), c0.repeat(len(m))))
        md = torch.cat(outs)
    return ((m0 - md) / (abs(m0 - mf) + 0.5)).cpu().numpy().astype(np.float32), float(m0), float(mf)


def edge_shifts(r):
    out = []
    for ei in range(4):
        for d in (-2, 2):
            cand = [r[0], r[1], r[2], r[3]]
            cand[ei] = int(np.clip(cand[ei] + d, 0, 64))
            cand = tuple(cand)
            if cand[2] - cand[0] >= 6 and cand[3] - cand[1] >= 6:
                out.append(cand)
            else:
                out.append(None)
    return out


def noise_edge_jump(e, r):
    x0, y0, x1, y1 = r
    feats = []
    specs = [
        (e[y0:y1, x0:min(x0 + 2, x1)], e[y0:y1, max(x0 - 2, 0):x0]),
        (e[y0:min(y0 + 2, y1), x0:x1], e[max(y0 - 2, 0):y0, x0:x1]),
        (e[y0:y1, max(x1 - 2, x0):x1], e[y0:y1, x1:min(x1 + 2, 64)]),
        (e[max(y1 - 2, y0):y1, x0:x1], e[y1:min(y1 + 2, 64), x0:x1]),
    ]
    for inside, outside in specs:
        ii = float(inside.mean()) if inside.size else 0.0
        oo = float(outside.mean()) if outside.size else 0.0
        feats += [ii, oo, ii - oo]
    return feats


def grad_maps(bb, u8, dev):
    with torch.enable_grad():
        x = to_t(u8[None]).to(dev).requires_grad_(True)
        z = bb(x)
        top = z.topk(2, dim=1).values
        g2 = torch.autograd.grad((top[:, 0] - top[:, 1]).sum(), x)[0].detach()
    xd = x.detach()
    rsd = xd - F.avg_pool2d(F.pad(xd, (1,) * 4, mode='reflect'), 3, stride=1)
    return rsd, g2


def corr_mask(rsd, g2, mask):
    num = (rsd * g2 * mask).sum()
    a = (rsd * rsd * mask).sum().clamp_min(1e-12).sqrt()
    b = (g2 * g2 * mask).sum().clamp_min(1e-12).sqrt()
    return float((num / (a * b + 1e-12)).cpu())


def unet_mass(P, r):
    x0, y0, x1, y1 = r
    a = max((x1 - x0) * (y1 - y0), 1)
    X0, Y0, X1, Y1 = max(x0 - 4, 0), max(y0 - 4, 0), min(x1 + 4, 64), min(y1 + 4, 64)
    ring_a = max((X1 - X0) * (Y1 - Y0) - a, 1)
    out = []
    for ch in (1, 2):
        s = float(P[ch, y0:y1, x0:x1].sum())
        so = float(P[ch, Y0:Y1, X0:X1].sum()) - s
        out += [s / a, so / ring_a, s]
    out += [a / 4096., (x1 - x0) / max(y1 - y0, 1), (x1 - x0) / 64., (y1 - y0) / 64.]
    return out


def estimate_jpeg_q(u8):
    e90 = np.abs(u8.astype(np.float32) - jpeg_u8(u8, 90).astype(np.float32)).mean()
    e80 = np.abs(u8.astype(np.float32) - jpeg_u8(u8, 80).astype(np.float32)).mean()
    return [float(e90), float(e80), float(e80 / (e90 + 1e-6))]


def feats_for_image(bb, u8, P, C, rects, starts, dev):
    """Zwraca (n_var, n_feat*2) — cechy + roznica vs wariant startowy (rects[0]).
    Profil krawedzi liczony tylko dla startu (cecha obrazka/startu), zeby nie mnozyc forwardow.
    """
    x = to_t(u8[None]).to(dev)
    f30 = to_t(jpeg_u8(u8, 30)[None]).to(dev)
    f50 = to_t(jpeg_u8(u8, 50)[None]).to(dev)
    r0 = starts[0]
    extras = [c for c in edge_shifts(r0) if c is not None]
    all_r = rects + extras
    idx = {r: i for i, r in enumerate(all_r)}
    dr30, m0, mf30 = margins_many(bb, x, f30, all_r, dev, 'remove')
    dr50, _, mf50 = margins_many(bb, x, f50, rects, dev, 'remove')
    dk30, _, _ = margins_many(bb, x, f30, rects, dev, 'keep')
    dk50, _, _ = margins_many(bb, x, f50, rects, dev, 'keep')

    # profil krawedzi startu (8 liczb) — wspolny dla wszystkich wariantow
    base_d = float(dr30[idx[r0]]) if r0 in idx else float(dr30[0])
    ep0 = []
    for c in edge_shifts(r0):
        ep0.append(0.0 if c is None else float(dr30[idx[c]] - base_d))

    e_noise = np.abs(u8.astype(np.float32) / 255. - median_filter(u8.astype(np.float32) / 255., size=(3, 3, 1))).mean(-1)
    rsd, g2 = grad_maps(bb, u8, dev)
    jpeg_f = estimate_jpeg_q(u8)
    cz = np.clip(C.mean(0).astype(np.float32), 0, None); cz /= cz.max() + 1e-9
    pc = P[1].astype(np.float32) * cz
    I = integral(pc.astype(np.float64)); A = float(pc.sum())
    P32 = P.astype(np.float32)

    rows = []
    for j, r in enumerate(rects):
        M = torch.zeros(1, 1, 64, 64, device=dev)
        M[:, :, r[1]:r[3], r[0]:r[2]] = 1
        Rm = torch.zeros_like(M)
        Rm[:, :, max(r[1] - 4, 0):min(r[3] + 4, 64), max(r[0] - 4, 0):min(r[2] + 4, 64)] = 1
        Rm = Rm - M
        gc = [corr_mask(rsd, g2, M), corr_mask(rsd, g2, Rm)]
        s = I[r[3], r[2]] - I[r[1], r[2]] - I[r[3], r[0]] + I[r[1], r[0]]
        ar = (r[2] - r[0]) * (r[3] - r[1])
        # delta krawedzi wariantu vs start (geometryczna)
        d_edge = [r[k] - r0[k] for k in range(4)]
        f = [float(dr30[idx[r]]), float(dr50[j]), float(dk30[j]), float(dk50[j]), m0, mf30, mf50]
        f += ep0 + d_edge
        f += noise_edge_jump(e_noise, r)
        f += gc
        f += unet_mass(P32, r)
        f += [s / (ar + A - s + 1e-9), float(r == r0)]
        f += jpeg_f
        rows.append(np.asarray(f, np.float32))
    F = np.stack(rows)
    return np.concatenate([F, F - F[0:1]], 1)


def load_train_rows_like_oof():
    real = E.read_train(D)
    rows = list(real)
    for k, (d, mx) in enumerate(zip(['synth_v2', 'synth_final'], [6000, 6000])):
        if os.path.exists(os.path.join(d, 'train_labels.csv')):
            ex = E.read_train(d)
            ex = [ex[i] for i in np.random.default_rng(k).permutation(len(ex))[:mx]]
            rows += ex
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--oof', default='cache_oof')
    ap.add_argument('--out', default='cache_rank')
    ap.add_argument('--max', type=int, default=0)
    ap.add_argument('--starts', type=int, default=2)
    ap.add_argument('--with8', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = 'cuda'
    bb = backbone(dev)
    done_path = os.path.join(a.out, f'{a.split}.done')
    start = int(open(done_path).read()) if os.path.exists(done_path) else 0
    cache_partial = os.path.join(a.out, f'{a.split}_partial.npz')

    if a.split == 'val':
        rows = train_rows()
        va = np.sort(np.random.default_rng(0).permutation(len(rows))[:500])
        idxs = [i for i in va if rows[i]['box'] != NONE]
        ids = [rows[i]['id'] for i in idxs]
        gts = [rows[i]['box'] for i in idxs]
        U = load_imgs(ids, 'train')
        Pv = np.load('cache_v2/unet_oof_val.npz')['P'].astype(np.float32)
        va_pos = {int(v): j for j, v in enumerate(va)}
        P = np.stack([Pv[va_pos[i]] for i in idxs])
        C = np.load('cache_v2/causal_val.npy').astype(np.float32)
        C = np.stack([C[va_pos[i]] for i in idxs])
    elif a.split == 'test':
        ids = test_ids(); gts = [NONE] * len(ids)
        U = load_imgs(ids, 'test')
        P = np.load(os.path.join(a.oof, 'P_test.npz'))['P'].astype(np.float32)
        C = np.load('cache_v2/causal_test.npy').astype(np.float32)
    else:
        meta = np.load(os.path.join(a.oof, 'meta.npz'), allow_pickle=True)
        va_real = set(meta['va_real'].tolist())
        rows = load_train_rows_like_oof()
        P_all = np.load(os.path.join(a.oof, 'P_train.f16.npy'), mmap_mode='r')
        C_all = np.load('cache_v3/causal_train.npy', mmap_mode='r')
        assert len(rows) == len(P_all)
        idxs = [i for i in range(len(rows)) if i not in va_real and rows[i]['box'] != NONE]
        ids = [rows[i]['id'] for i in idxs]
        gts = [rows[i]['box'] for i in idxs]
        U = np.stack([E.load_u8(rows[i]['root'], 'train', rows[i]['id']) for i in idxs])
        P = np.asarray(P_all[idxs], np.float32)
        C = np.asarray(C_all[idxs], np.float32)

    if a.max:
        U, P, C, gts, ids = U[:a.max], P[:a.max], C[:a.max], gts[:a.max], ids[:a.max]
    N = len(U)

    if start > 0 and os.path.exists(cache_partial):
        z = np.load(cache_partial, allow_pickle=True)
        all_F, all_R, all_y, all_g = list(z['F']), list(z['R']), list(z['y']), list(z['gid'])
        assert int(z['n']) == start
    else:
        all_F, all_R, all_y, all_g, start = [], [], [], [], 0

    t0 = time.time()
    for n in range(start, N):
        u, p, c, gt = U[n], P[n], C[n], gts[n]
        cz = np.clip(c.mean(0), 0, None); cz = cz / (cz.max() + 1e-9)
        starts = top_starts(p[1] * cz, a.starts)
        R, S0 = gather_rects(starts, with8=a.with8)
        F = feats_for_image(bb, u, p, c, R, S0, dev)
        y = np.array([punkty(gt, r) if gt != NONE else 0.0 for r in R], np.float32)
        all_F.append(F); all_R.append(np.array(R, np.int16)); all_y.append(y); all_g.append(n)
        if (n + 1) % 10 == 0 or n + 1 == N:
            np.savez_compressed(cache_partial, F=np.array(all_F, dtype=object), R=np.array(all_R, dtype=object),
                                y=np.array(all_y, dtype=object), gid=np.array(all_g), n=n + 1)
            open(done_path, 'w').write(str(n + 1))
            print(f'{a.split} {n + 1}/{N}  vars~{len(R)}  {time.time() - t0:.0f}s', flush=True)
    np.savez_compressed(os.path.join(a.out, f'{a.split}_feats.npz'),
                        F=np.array(all_F, dtype=object), R=np.array(all_R, dtype=object),
                        y=np.array(all_y, dtype=object), gid=np.array(all_g), ids=np.array(ids))
    print('gotowe', a.split, N)


if __name__ == '__main__':
    main()
