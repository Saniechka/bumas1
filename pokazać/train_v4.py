"""
train_v4.py - U-Net v3 + uczenie KRAWEDZI ramki.

Zmiany wzgledem train_v3.py (te same dane, cechy i mapy przyczynowe z --cache):
  1. Druga "glowa" sieci: kazdy piksel latki przewiduje odleglosc do lewej, gornej, prawej
     i dolnej krawedzi (jak detektor FCOS). Ramka = glosowanie setek pikseli -> dokladniejsze
     krawedzie niz dopasowanie prostokata do mapy prawdopodobienstwa.
  2. Wieksza waga bledow na pikselach przy krawedzi latki (--edge_w).
  3. Prawdziwe obrazki czesciej w treningu (--real_rep), bo test przypomina wlasnie je.
  4. Start z gotowych wag v3 (--init), wiec wystarczy kilkanascie-kilkadziesiat epok.

Wynik:
  cache_v2/unet4_val.npz, unet4_test.npz   - mapy jak z v3 (do solve_v2.py --unet ...,unet4)
  cache_v2/unet4_box_val.npy, _test.npy    - ramki z glosowania (do apply_box.py)

    python train_v4.py --synth ../bumas1/synth_v2 --synth_max 50000 --cache cache_v3_full \
                       --init cache_v3_full/unet3_1.pt --epochs 25
"""
import argparse, csv, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'exp'))
import ensemble4 as E
import train_v3 as T3
from common import punkty, NONE, best_rect, D
from causal_cache import compute_causal


# =============================================================== siec
class BoxUNet(E.PatchUNet):
    def __init__(self, cin, w):
        super().__init__(cin=cin, w=w)
        self.box = nn.Conv2d(w, 4, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        e4 = self.e4(F.avg_pool2d(e3, 2))
        g = self.head(e4.mean((2, 3))).squeeze(1)
        up = lambda a: F.interpolate(a, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = self.d3(torch.cat([up(e4), e3], 1))
        d2 = self.d2(torch.cat([up(d3), e2], 1))
        d1 = self.d1(torch.cat([up(d2), e1], 1))
        dist = F.softplus(self.box(d1).float()) * 16.0      # odleglosci w pikselach (l, t, r, b)
        return self.out(d1), g, dist


# =============================================================== geometria
def d4_box(b, k):
    """Ramka po tej samej operacji co E.d4_np/E.d4_t (najpierw transpozycja, potem odbicia)."""
    x0, y0, x1, y1 = b
    if k & 4: x0, y0, x1, y1 = y0, x0, y1, x1
    if k & 1: x0, x1 = 64 - x1, 64 - x0
    if k & 2: y0, y1 = 64 - y1, 64 - y0
    return (x0, y0, x1, y1)


GRID = (torch.arange(64).float() + 0.5)


def box_targets(boxes, has):
    """boxes (B,4) float, has (B,) -> odleglosci (B,4,64,64) i maska pikseli latki (B,64,64)."""
    g = GRID.to(boxes.device)
    X = g.view(1, 1, 64).expand(len(boxes), 64, 64)
    Y = g.view(1, 64, 1).expand(len(boxes), 64, 64)
    x0, y0, x1, y1 = [boxes[:, i].view(-1, 1, 1) for i in range(4)]
    T = torch.stack([X - x0, Y - y0, x1 - X, y1 - Y], 1)
    inside = (T.min(1).values > 0) & (has.view(-1, 1, 1) > 0)
    return T, inside


def iou_loss(pred, T, inside):
    if not inside.any():
        return pred.sum() * 0
    pl, pt, pr, pb = pred.unbind(1); tl, tt, tr, tb = T.unbind(1)
    pa = (pl + pr) * (pt + pb); ta = (tl + tr) * (tt + tb)
    inter = (torch.min(pl, tl) + torch.min(pr, tr)) * (torch.min(pt, tt) + torch.min(pb, tb))
    iou = inter / (pa + ta - inter + 1e-6)
    return (1 - iou)[inside].mean()


def dist_to_coords(dist):
    """(B,4,64,64) odleglosci -> (B,4,64,64) bezwzgledne wspolrzedne ramki x0,y0,x1,y1."""
    g = GRID.to(dist.device)
    X = g.view(1, 1, 64); Y = g.view(1, 64, 1)
    return torch.stack([X - dist[:, 0], Y - dist[:, 1], X + dist[:, 2], Y + dist[:, 3]], 1)


def coords_inv(c, k):
    """Wspolrzedne zapisane w ukladzie po d4(k) -> uklad oryginalny (odwrotnosc d4_box)."""
    x0, y0, x1, y1 = c.unbind(1)
    if k & 2: y0, y1 = 64 - y1, 64 - y0
    if k & 1: x0, x1 = 64 - x1, 64 - x0
    if k & 4: x0, y0, x1, y1 = y0, x0, y1, x1
    return torch.stack([x0, y0, x1, y1], 1)


def vote(p1, coords, thr=0.3):
    """p1 (64,64) prawdop. latki, coords (4,64,64) -> ramka z glosowania lub None."""
    l = coords[0]; t = coords[1]
    w = np.where(p1 > thr, p1 ** 2, 0.0)
    if w.sum() < 1e-3:
        return None
    b = [(coords[i] * w).sum() / w.sum() for i in range(4)]
    x0, y0, x1, y1 = [int(round(float(v))) for v in b]
    x0, y0 = max(0, min(x0, 60)), max(0, min(y0, 60))
    x1, y1 = max(x0 + 4, min(x1, 64)), max(y0 + 4, min(y1, 64))
    return (x0, y0, x1, y1)


# =============================================================== trening
def train(a, dev, X, rows, tr_idx, n_real, mu, sd, out_path, seed, init, eps=None):
    torch.manual_seed(seed)
    net = BoxUNet(cin=X.shape[1], w=a.width).to(dev)
    if init and os.path.exists(init):
        miss = net.load_state_dict(torch.load(init, map_location=dev, weights_only=True), strict=False)
        print(f'   start z wag {init} (nowe warstwy: {miss.missing_keys})', flush=True)
    tr_real = tr_idx[tr_idx < n_real]
    parts = [tr_idx] + [tr_real] * (a.real_rep - 1)
    if eps is not None and a.eps_boost > 1:
        weak = tr_idx[(eps[tr_idx] > 0) & (eps[tr_idx] <= a.eps_max)]
        parts += [weak] * (a.eps_boost - 1)
        print(f'   slabe latki (eps <= {a.eps_max}/255): {len(weak)} obrazkow, pokazywane x{a.eps_boost}', flush=True)
    pool = np.concatenate(parts)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = max(1, len(pool) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps, pct_start=0.15)
    wcls = torch.tensor([0.25, 1.0, 0.6], device=dev)
    masks = np.stack([E.label_mask(r) for r in rows]).astype(np.uint8)
    boxes = np.array([r['box'] for r in rows], np.float32)
    has = np.array([r['box'] != NONE for r in rows], np.float32)
    mu_t, sd_t = torch.from_numpy(mu).to(dev), torch.from_numpy(sd).to(dev)
    rng = np.random.default_rng(seed)
    print(f'   probek na epoke: {len(pool)} (prawdziwe x{a.real_rep}), krokow {steps}', flush=True)
    for ep in range(a.epochs):
        net.train(); perm = rng.permutation(pool); tot = tb = n = 0; t = time.time()
        for s in range(steps):
            b = perm[s * a.bs:(s + 1) * a.bs]
            o = np.argsort(b); b = b[o]
            Xb = X[b]; M = masks[b].copy(); Bx = boxes[b].copy()
            for j in range(len(b)):
                k = int(rng.integers(8))
                Xb[j] = E.d4_np(Xb[j], k).copy(); M[j] = E.d4_np(M[j], k).copy()
                if has[b[j]]:
                    Bx[j] = d4_box(tuple(Bx[j]), k)
            x = (torch.from_numpy(np.ascontiguousarray(Xb)).to(dev) - mu_t) / sd_t
            m = torch.from_numpy(np.ascontiguousarray(M).astype(np.int64)).to(dev)
            h = torch.from_numpy(has[b]).to(dev)
            bt = torch.from_numpy(Bx).to(dev)
            T, inside = box_targets(bt, h)
            edge = inside & (T.min(1).values < 2.5)                     # 2 px przy krawedzi latki (wewnatrz)
            ring = (~inside) & (h.view(-1, 1, 1) > 0) & (T.min(1).values > -2.5) & \
                   (T.min(1).values <= 0) & (T[:, [0, 2]].min(1).values > -2.5) & (T[:, [1, 3]].min(1).values > -2.5)
            wmap = 1.0 + a.edge_w * (edge | ring).float()
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev != 'cpu')):
                logit, g, dist = net(x)
            logit, g = logit.float(), g.float()
            ce = (F.cross_entropy(logit, m, weight=wcls, reduction='none') * wmap).mean()
            lb = iou_loss(dist, T, inside)
            loss = ce + 0.5 * E.soft_dice(logit.softmax(1)[:, 1], (m == 1).float()) \
                + 0.3 * F.binary_cross_entropy_with_logits(g, h) + a.box_w * lb
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(b); tb += lb.item() * len(b); n += len(b)
        print(f'   [seed {seed}] epoka {ep + 1}/{a.epochs}  strata {tot / n:.4f}  (krawedzie {tb / n:.4f})'
              f'  {time.time() - t:.0f}s', flush=True)
    torch.save(net.state_dict(), out_path)
    return net


@torch.no_grad()
def infer(nets, dev, X, mu, sd, bs=64):
    """-> P (N,3,64,64), G (N,), ramki z glosowania (N,4) (-1 gdy brak)."""
    mu_t, sd_t = torch.from_numpy(mu).to(dev), torch.from_numpy(sd).to(dev)
    N = len(X)
    P = np.zeros((N, 3, 64, 64), np.float32); G = np.zeros(N, np.float32)
    B = np.full((N, 4), -1, np.int32)
    for s in range(0, N, bs):
        idx = np.arange(s, min(s + bs, N))
        x = (torch.from_numpy(X[idx]).to(dev) - mu_t) / sd_t
        accP = 0; accG = 0; accC = 0; cnt = 0
        for net in nets:
            net.eval()
            for k in range(8):
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev != 'cpu')):
                    lg, g, dist = net(E.d4_t(x, k))
                accP = accP + E.d4_t_inv(lg.float().softmax(1), k)
                accG = accG + g.float()
                c = E.d4_t_inv(dist_to_coords(dist.float()), k)
                accC = accC + coords_inv(c, k)
                cnt += 1
        p = (accP / cnt).cpu().numpy(); c = (accC / cnt).cpu().numpy()
        P[idx] = p; G[idx] = (accG / cnt).cpu().numpy()
        for j, i in enumerate(idx):
            v = vote(p[j, 1], c[j])
            if v is not None:
                B[i] = v
    return P, G, B


def report(name, P, B, gt, C):
    has = [i for i, g in enumerate(gt) if g != NONE]
    seg, segc, reg, avg = [], [], [], []
    for i in has:
        cz = np.clip(np.asarray(C[i], np.float32).mean(0), 0, None); cz /= cz.max() + 1e-9
        b1 = best_rect(P[i, 1])[0]; b2 = best_rect(P[i, 1] * cz)[0]
        seg.append(punkty(gt[i], b1)); segc.append(punkty(gt[i], b2))
        if B[i][0] >= 0:
            r = tuple(int(v) for v in B[i]); reg.append(punkty(gt[i], r))
            m = tuple(int(round((u + v) / 2)) for u, v in zip(b2, r)); avg.append(punkty(gt[i], m))
        else:
            reg.append(punkty(gt[i], b2)); avg.append(punkty(gt[i], b2))
    print(f'   {name:22s} IoU: mapa {np.mean(seg):.3f} | mapa x przyczynowa {np.mean(segc):.3f}'
          f' | KRAWEDZIE (glosowanie) {np.mean(reg):.3f} | srednia obu {np.mean(avg):.3f}', flush=True)


# =============================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synth', default='../bumas1/synth_v2')
    ap.add_argument('--synth_max', default='50000')
    ap.add_argument('--synth_val', type=int, default=500)
    ap.add_argument('--cache', default='cache_v3_full', help='ten sam folder co w train_v3 (cechy i mapy)')
    ap.add_argument('--init', default='', help='wagi v3 na start, np. cache_v3_full/unet3_1.pt')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--width', type=int, default=96)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--real_rep', type=int, default=3, help='ile razy czesciej pokazywac prawdziwe obrazki')
    ap.add_argument('--edge_w', type=float, default=2.0, help='dodatkowa waga pikseli przy krawedzi')
    ap.add_argument('--box_w', type=float, default=1.0, help='waga straty krawedzi (IoU)')
    ap.add_argument('--tag', default='unet4')
    ap.add_argument('--eps_boost', type=int, default=1, help='ile razy czesciej pokazywac slabe latki (eps <= --eps_max)')
    ap.add_argument('--eps_max', type=int, default=4)
    ap.add_argument('--full', action='store_true',
                    help='ucz takze na 500 obrazkach walidacji (na koniec; walidacja jest wtedy zawyzona)')
    a = ap.parse_args()
    dev = 'cuda'
    torch.backends.cudnn.benchmark = True
    bb = E.load_backbone(D, dev)

    # ---- dane: dokladnie jak w train_v3.py (ta sama kolejnosc -> ten sam cache)
    real = E.read_train(D)
    perm = np.random.default_rng(0).permutation(len(real))
    va_real = np.sort(perm[:500])
    rows = list(real); syn_val = []
    for k, (d, mx) in enumerate(zip(a.synth.split(','), map(int, a.synth_max.split(',')))):
        ex = E.read_train(d)
        ex = [ex[i] for i in np.random.default_rng(k).permutation(len(ex))[:mx]]
        s0 = len(rows); rows += ex
        if not syn_val:
            syn_val = list(range(s0, s0 + min(a.synth_val, len(ex) // 5)))
        print(f'{d}: {len(ex)} obrazkow')
    syn_val = np.array(syn_val, int)
    va_all = np.concatenate([va_real, syn_val])
    tr_idx = np.setdiff1d(np.arange(len(rows)), syn_val if a.full else va_all)
    print(f'trening {len(tr_idx)}, walidacja: prawdziwa {len(va_real)}, syntetyczna {len(syn_val)}'
          + ('   (FULL: walidacja prawdziwa jest w treningu - jej wynik bedzie zawyzony)' if a.full else ''))
    # eps kazdego obrazka z komentarza w train_labels.csv (0 = brak latki)
    import re
    eps_of = {}
    for root in [D] + a.synth.split(','):
        for r in csv.DictReader(open(os.path.join(root, 'train_labels.csv'))):
            m = re.search(r'eps=(\d+)', r['comment'])
            eps_of[(root, r['id'])] = int(m.group(1)) if (m and r['comment'].startswith('ADV')) else 0
    eps = np.array([eps_of.get((r['root'], r['id']), 0) for r in rows])

    print('[1/3] cechy i mapy przyczynowe (z cache)')
    p_f = os.path.join(a.cache, 'f_train.npy')
    need = not (os.path.exists(p_f) and os.path.exists(p_f + '.logits.npy'))
    U = np.stack([E.load_u8(r['root'], 'train', r['id']) for r in rows]) if need else None
    f_tr, _ = T3.feats_cache(bb, dev, U, p_f)
    p_c = os.path.join(a.cache, 'causal_train.npy')
    if U is None and not (os.path.exists(p_c.replace('.npy', '.done'))
                          and int(open(p_c.replace('.npy', '.done')).read()) >= len(rows)):
        U = np.stack([E.load_u8(r['root'], 'train', r['id']) for r in rows])
    c_tr, _ = compute_causal(bb, U if U is not None else np.zeros((len(rows), 1)), p_c, dev, 'trening')
    assert len(f_tr) == len(rows) == len(c_tr), (len(f_tr), len(rows), len(c_tr), 'zly --synth_max albo --cache')
    ids = [r['id'] for r in csv.DictReader(open(os.path.join(D, 'test_ids.csv')))]
    p_ft = os.path.join(a.cache, 'f_test.npy')
    Ut = None if os.path.exists(p_ft) else T3.load_set(D, 'test', ids)
    f_te, _ = T3.feats_cache(bb, dev, Ut, p_ft)
    c_te, _ = compute_causal(bb, np.zeros((len(ids), 1)), 'cache_v2/causal_test.npy', dev, 'test')
    del U, Ut
    X, Xt = T3.Stack(f_tr, c_tr), T3.Stack(f_te, c_te)
    smp = X[np.setdiff1d(np.arange(len(rows)), va_all)[:400]]   # ta sama normalizacja co v3
    mu = smp.mean((0, 2, 3), dtype=np.float64).reshape(1, -1, 1, 1).astype(np.float32)
    sd = (smp.std((0, 2, 3), dtype=np.float64) + 1e-4).reshape(1, -1, 1, 1).astype(np.float32)
    n_real = len(real)

    print('[2/3] trening')
    nets = []
    for i, seed in enumerate(int(s) for s in a.seeds.split(',')):
        p = os.path.join(a.cache, f'{a.tag}_{i + 1}.pt')
        if os.path.exists(p):
            n = BoxUNet(cin=X.shape[1], w=a.width).to(dev)
            n.load_state_dict(torch.load(p, map_location=dev, weights_only=True)); nets.append(n)
            print(f'   wczytano {p}')
        else:
            init = a.init.split(',')[min(i, len(a.init.split(',')) - 1)] if a.init else ''
            nets.append(train(a, dev, X, rows, tr_idx, n_real, mu, sd, p, seed, init, eps))

    print('[3/3] walidacja i test')
    Pv, Gv, Bv = infer(nets, dev, X[va_real], mu, sd)
    report('walidacja prawdziwa', Pv, Bv, [rows[i]['box'] for i in va_real], c_tr[va_real])
    if len(syn_val):
        Ps, _, Bs = infer(nets, dev, X[syn_val], mu, sd)
        report('walidacja syntetyczna', Ps, Bs, [rows[i]['box'] for i in syn_val], c_tr[syn_val])
    np.savez_compressed(f'cache_v2/{a.tag}_val.npz', P=Pv.astype(np.float16), G=Gv)
    np.save(f'cache_v2/{a.tag}_box_val.npy', Bv)
    Pt = np.zeros((len(Xt), 3, 64, 64), np.float16); Gt = np.zeros(len(Xt), np.float32)
    Bt = np.full((len(Xt), 4), -1, np.int32)
    for s in range(0, len(Xt), 500):
        b = np.arange(s, min(s + 500, len(Xt)))
        p, g, bx = infer(nets, dev, Xt[b], mu, sd); Pt[b] = p.astype(np.float16); Gt[b] = g; Bt[b] = bx
    np.savez_compressed(f'cache_v2/{a.tag}_test.npz', P=Pt, G=Gt)
    np.save(f'cache_v2/{a.tag}_box_test.npy', Bt)
    print(f'zapisano cache_v2/{a.tag}_val/test.npz i {a.tag}_box_val/test.npy')


if __name__ == '__main__':
    main()
