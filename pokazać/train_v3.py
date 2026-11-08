"""
P1 v3: U-Net uczony na 24 cechach z solve.py + 2 kanalach mapy przyczynowej (JPEG q30/q50),
na prawdziwym treningu i danych z generate_v2.py. Wynik zapisuje w formacie solve_v2.py.

    python generate_v2.py --data data/p1_student --out synth_v2 --per_base 8
    python train_v3.py --synth synth_v2,synth_final --synth_max 6000,6000
    python solve_v2.py --unet unet3 --out sub_v3.csv

Wszystkie etapy maja cache w cache_v3/ i da sie je wznowic po przerwaniu.
"""
import argparse, csv, os, re, sys, time
import numpy as np
import torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'exp'))
import ensemble4 as E
from common import punkty, NONE, best_rect, D
from causal_cache import compute_causal, QS

NC = len(QS)


def load_set(root, split, ids):
    return np.stack([E.load_u8(root, split, i) for i in ids])


def feats_cache(bb, dev, imgs, path):
    f, lg = E.cache_split(bb, dev, imgs, path, 128)
    return f, lg


class Stack:
    """Leniwe sklejenie cech (N,24,..) i znormalizowanych map przyczynowych (N,NC,..)."""
    def __init__(self, f, c):
        self.f, self.c = f, c
        self.shape = (len(f), f.shape[1] + c.shape[1], 64, 64)

    def __len__(self):
        return len(self.f)

    def __getitem__(self, b):
        c = np.clip(np.asarray(self.c[b], np.float32), 0, None)
        c = c / (c.max((-2, -1), keepdims=True) + 1e-6)
        return np.concatenate([np.asarray(self.f[b], np.float32), c], 1)


def train(a, dev, X, rows, tr_idx, mu, sd, out_path, seed):
    torch.manual_seed(seed)
    net = E.PatchUNet(cin=X.shape[1], w=a.width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = max(1, len(tr_idx) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps)
    wcls = torch.tensor([0.25, 1.0, 0.6], device=dev)
    masks = np.stack([E.label_mask(r) for r in rows]).astype(np.uint8)
    has = np.array([r['box'] != NONE for r in rows], np.float32)
    mu_t, sd_t = torch.from_numpy(mu).to(dev), torch.from_numpy(sd).to(dev)
    rng = np.random.default_rng(seed)
    scaler_ctx = dict(device_type='cuda', dtype=torch.bfloat16, enabled=(dev != 'cpu'))
    for ep in range(a.epochs):
        net.train(); perm = rng.permutation(tr_idx); tot = n = 0; t = time.time()
        for s in range(steps):
            b = np.sort(perm[s * a.bs:(s + 1) * a.bs])
            Xb = X[b]; M = masks[b].copy()
            for j in range(len(b)):
                k = int(rng.integers(8))
                Xb[j] = E.d4_np(Xb[j], k).copy(); M[j] = E.d4_np(M[j], k).copy()
            x = (torch.from_numpy(np.ascontiguousarray(Xb)).to(dev) - mu_t) / sd_t
            m = torch.from_numpy(np.ascontiguousarray(M).astype(np.int64)).to(dev)
            h = torch.from_numpy(has[b]).to(dev)
            with torch.autocast(**scaler_ctx):
                logit, g = net(x)
            logit, g = logit.float(), g.float()
            loss = F.cross_entropy(logit, m, weight=wcls) \
                + 0.5 * E.soft_dice(logit.softmax(1)[:, 1], (m == 1).float()) \
                + 0.3 * F.binary_cross_entropy_with_logits(g, h)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(b); n += len(b)
        print(f'   [seed {seed}] epoka {ep + 1}/{a.epochs}  strata {tot / n:.4f}  {time.time() - t:.0f}s', flush=True)
    torch.save(net.state_dict(), out_path)
    return net


@torch.no_grad()
def infer(nets, dev, X, mu, sd, bs=64):
    mu_t, sd_t = torch.from_numpy(mu).to(dev), torch.from_numpy(sd).to(dev)
    P = np.zeros((len(X), 3, 64, 64), np.float32); G = np.zeros(len(X), np.float32)
    for net in nets:
        net.eval()
        for s in range(0, len(X), bs):
            x = (torch.from_numpy(X[np.arange(s, min(s + bs, len(X)))]).to(dev) - mu_t) / sd_t
            acc = 0; gacc = 0
            for k in range(8):
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev != 'cpu')):
                    lg, g = net(E.d4_t(x, k))
                acc = acc + E.d4_t_inv(lg.float().softmax(1), k); gacc = gacc + g.float()
            P[s:s + len(x)] += (acc / 8).cpu().numpy() / len(nets); G[s:s + len(x)] += (gacc / 8).cpu().numpy() / len(nets)
    return P, G


def report(name, P, gt, C=None):
    has = [g != NONE for g in gt]
    iou = [punkty(g, best_rect(p[1])[0]) for p, g, h in zip(P, gt, has) if h]
    msg = f'   {name:22s} IoU(U-Net) {np.mean(iou):.3f}'
    if C is not None:
        iou2 = []
        for p, c, g, h in zip(P, C, gt, has):
            if h:
                cz = np.clip(np.asarray(c, np.float32).mean(0), 0, None); cz /= cz.max() + 1e-9
                iou2.append(punkty(g, best_rect(p[1] * cz)[0]))
        msg += f'   IoU(U-Net x przyczynowa) {np.mean(iou2):.3f}'
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synth', default='synth_v2,synth_final', help='foldery z generate*.py, po przecinku')
    ap.add_argument('--synth_max', default='6000,6000', help='ile obrazkow wziac z kazdego folderu')
    ap.add_argument('--synth_val', type=int, default=500, help='ile obrazkow z PIERWSZEGO folderu odlozyc do walidacji')
    ap.add_argument('--cache', default='cache_v3')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    dev = 'cuda'
    os.makedirs(a.cache, exist_ok=True); os.makedirs('cache_v2', exist_ok=True)
    if a.smoke:
        a.epochs, a.width, a.cache = 1, 16, a.cache + '_smoke'
        os.makedirs(a.cache, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    bb = E.load_backbone(D, dev)

    # ---- dane: prawdziwy trening (te same 500 odlozone co w solve/ensemble4) + syntetyczne
    real = E.read_train(D)
    perm = np.random.default_rng(0).permutation(len(real))
    va_real = np.sort(perm[:500])
    rows = list(real)
    syn_val = []
    for k, (d, mx) in enumerate(zip(a.synth.split(','), map(int, a.synth_max.split(',')))):
        if not os.path.exists(os.path.join(d, 'train_labels.csv')):
            print(f'pomijam {d} (brak)'); continue
        ex = E.read_train(d)
        ex = [ex[i] for i in np.random.default_rng(k).permutation(len(ex))[:mx]]
        if a.smoke:
            ex = ex[:100]
        s0 = len(rows); rows += ex
        if not syn_val:
            syn_val = list(range(s0, s0 + min(a.synth_val, len(ex) // 5)))
        print(f'{d}: {len(ex)} obrazkow')
    if a.smoke:
        keep = list(va_real[:50]) + list(perm[500:600]) + list(range(len(real), len(rows)))
        rows = [rows[i] for i in keep]; va_real = np.arange(50)
        syn_val = list(range(150, 150 + 20))
    syn_val = np.array(syn_val, int)
    va_all = np.concatenate([va_real, syn_val])
    tr_idx = np.setdiff1d(np.arange(len(rows)), va_all)
    print(f'trening {len(tr_idx)}, walidacja: prawdziwa {len(va_real)}, syntetyczna {len(syn_val)}')

    # ---- cechy i mapy przyczynowe (cache)
    print('[1/3] cechy i mapy przyczynowe')
    U = np.stack([E.load_u8(r['root'], 'train', r['id']) for r in rows])
    f_tr, _ = feats_cache(bb, dev, U, os.path.join(a.cache, 'f_train.npy'))
    c_tr, s_tr = compute_causal(bb, U, os.path.join(a.cache, 'causal_train.npy'), dev, 'trening')
    ids = [r['id'] for r in csv.DictReader(open(os.path.join(D, 'test_ids.csv')))]
    if a.smoke:
        ids = ids[:50]
    Ut = load_set(D, 'test', ids)
    f_te, _ = feats_cache(bb, dev, Ut, os.path.join(a.cache, 'f_test.npy'))
    pc = 'cache_v2/causal_test.npy' if not a.smoke else os.path.join(a.cache, 'causal_test.npy')
    c_te, _ = compute_causal(bb, Ut, pc, dev, 'test')
    del U, Ut
    X, Xt = Stack(f_tr, c_tr), Stack(f_te, c_te)
    smp = X[tr_idx[:400]]
    mu = smp.mean((0, 2, 3), dtype=np.float64).reshape(1, -1, 1, 1).astype(np.float32)
    sd = (smp.std((0, 2, 3), dtype=np.float64) + 1e-4).reshape(1, -1, 1, 1).astype(np.float32)

    # ---- trening
    print('[2/3] trening')
    nets = []
    for i, seed in enumerate(int(s) for s in a.seeds.split(',')):
        p = os.path.join(a.cache, f'unet3_{i + 1}.pt')
        if os.path.exists(p):
            n = E.PatchUNet(cin=X.shape[1], w=a.width).to(dev)
            n.load_state_dict(torch.load(p, map_location=dev, weights_only=True)); nets.append(n)
            print(f'   wczytano {p}')
        else:
            nets.append(train(a, dev, X, rows, tr_idx, mu, sd, p, seed))

    # ---- walidacja i zapis map dla solve_v2.py
    print('[3/3] walidacja i test')
    Pv, Gv = infer(nets, dev, X[va_real], mu, sd)
    report('walidacja prawdziwa', Pv, [rows[i]['box'] for i in va_real], c_tr[va_real])
    if len(syn_val):
        Ps, _ = infer(nets, dev, X[syn_val], mu, sd)
        report('walidacja syntetyczna', Ps, [rows[i]['box'] for i in syn_val], c_tr[syn_val])
    tag = 'unet3' if not a.smoke else 'unet3_smoke'
    np.savez_compressed(f'cache_v2/{tag}_val.npz', P=Pv.astype(np.float16), G=Gv)
    Pt = np.zeros((len(Xt), 3, 64, 64), np.float16); Gt = np.zeros(len(Xt), np.float32)
    for s in range(0, len(Xt), 500):
        b = np.arange(s, min(s + 500, len(Xt)))
        p, g = infer(nets, dev, Xt[b], mu, sd); Pt[b] = p.astype(np.float16); Gt[b] = g
    np.savez_compressed(f'cache_v2/{tag}_test.npz', P=Pt, G=Gt)
    print(f'zapisano cache_v2/{tag}_val.npz i cache_v2/{tag}_test.npz -> python solve_v2.py --unet {tag}')


if __name__ == '__main__':
    main()
