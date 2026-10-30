"""
P1: Ukryta łatka — Ensemble 4 modeli U-Net.

Trenuje 4 niezależne modele z różnymi seedami, uśrednia ich predykcje,
generuje submission.csv.

    pip install torch numpy pillow
    python ensemble4.py --data /sciezka/do/p1_student
    python ensemble4.py --data /sciezka/do/p1_student --extra synth_final
    python ensemble4.py --data /sciezka/do/p1_student --smoke   # szybki test

Na 32 GB VRAM: każdy model ~5-10 min (RTX 5090), łącznie ~40 min.
"""
import argparse, csv, io, os, re, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

NONE = (0, 0, 0, 0)
NCH = 24
SEEDS = [42, 123, 777, 2026]


# =============================================================== ocena
def punkty(prawda, odp):
    if prawda == NONE:
        return 1.0 if odp == NONE else 0.0
    if odp == NONE:
        return 0.0
    w = max(0, min(prawda[2], odp[2]) - max(prawda[0], odp[0]))
    h = max(0, min(prawda[3], odp[3]) - max(prawda[1], odp[1]))
    wsp = w * h
    pole = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    return wsp / (pole(prawda) + pole(odp) - wsp)


# =============================================================== dane
def read_train(data):
    rows = list(csv.DictReader(open(os.path.join(data, 'train_labels.csv'))))
    wnids = [l.strip() for l in open(os.path.join(data, 'model', 'classes.txt')) if l.strip()]
    idx = {w: i for i, w in enumerate(wnids)}
    out = []
    for r in rows:
        b = tuple(int(r[k]) for k in ('x_min', 'y_min', 'x_max', 'y_max'))
        dec = [tuple(int(g) for g in m.groups())
               for m in re.finditer(r'at \[(\d+),(\d+),(\d+),(\d+)\]', r['comment'])]
        m = re.search(r'true=(n\d+)', r['comment'])
        out.append(dict(id=r['id'], box=b, decoys=dec, root=data,
                        true=idx.get(m.group(1), -1) if m else -1))
    return out


def read_test(data):
    return [r['id'] for r in csv.DictReader(open(os.path.join(data, 'test_ids.csv')))]


def load_u8(data, split, iid):
    return np.asarray(Image.open(os.path.join(data, split, 'images', iid + '.png')).convert('RGB'),
                      np.uint8)


def rejpeg(u8, q):
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, 'JPEG', quality=int(q))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert('RGB'), np.uint8)


def label_mask(r):
    m = np.zeros((64, 64), np.int64)
    for d in r['decoys']:
        m[d[1]:d[3], d[0]:d[2]] = 2
    b = r['box']
    if b != NONE:
        m[b[1]:b[3], b[0]:b[2]] = 1
    return m


# =============================================================== backbone
def load_backbone(data, dev):
    sys.path.insert(0, os.path.join(data, 'model'))
    from resnet import resnet18_tin
    m = resnet18_tin()
    m.load_state_dict(torch.load(os.path.join(data, 'model', 'resnet18_tin.pt'),
                                 map_location='cpu', weights_only=True))
    m.eval().to(dev)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# =============================================================== cechy
KB = torch.tensor([[-1., 2., -1.], [2., -4., 2.], [-1., 2., -1.]]) / 4.0


def _box(a, k):
    return F.avg_pool2d(F.pad(a, (k // 2,) * 4, mode='reflect'), k, stride=1)


def _median3(x):
    p = F.pad(x, (1, 1, 1, 1), mode='reflect')
    return p.unfold(2, 3, 1).unfold(3, 3, 1).reshape(*x.shape, 9).median(-1).values


def _dct8():
    n = torch.arange(8).float()
    m = torch.cos((2 * n + 1) * n.view(-1, 1) * torch.pi / 16) * (2.0 / 8) ** 0.5
    m[0] /= 2 ** 0.5
    return m


_D = _dct8()


def _block_dct(gray255):
    N = gray255.shape[0]
    b = gray255.reshape(N, 8, 8, 8, 8).permute(0, 1, 3, 2, 4)
    D = _D.to(gray255.device, gray255.dtype)
    ac = (D @ b @ D.t()).abs()
    hi = ac[..., 4:, 4:].mean((-2, -1))
    nz = (ac[..., 3:, 3:] > 1.0).float().mean((-2, -1))
    return F.interpolate(torch.stack([hi, nz], 1), scale_factor=8, mode='nearest')


def _corr(r, g, k):
    num = _box((r * g).sum(1, keepdim=True), k)
    a = _box((r * r).sum(1, keepdim=True), k)
    b = _box((g * g).sum(1, keepdim=True), k)
    return num / (a.sqrt() * b.sqrt() + 1e-12)


def build_features(bb, x):
    x = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        lg = bb(x)
        top = lg.topk(2, dim=1).values
        g1 = torch.autograd.grad(top[:, 0].sum(), x, retain_graph=True)[0]
        g2 = torch.autograd.grad((top[:, 0] - top[:, 1]).sum(), x)[0]
    x = x.detach()
    with torch.no_grad():
        gray = x.mean(1, keepdim=True)
        r = x - _median3(x)
        absr = _box(r.abs().mean(1, keepdim=True), 5)
        sg = (_box(gray * gray, 9) - _box(gray, 9) ** 2).clamp_min(1e-8).sqrt()
        kb = F.conv2d(F.pad(gray, (1,) * 4, mode='reflect'),
                      KB.to(x.device, x.dtype).view(1, 1, 3, 3))
        dct = _block_dct(gray * 255.0)
        rc = r - r.mean(1, keepdim=True)
        xc = _box(rc.abs().mean(1, keepdim=True), 5)
        sat = _box(((x < 0.02) | (x > 0.98)).float().mean(1, keepdim=True), 5)
        f = torch.cat([
            x, r * 20, absr * 20, absr / (sg + 0.01), _box(kb.abs(), 3) * 20, dct, xc * 20,
            g1 / (g1.abs().amax((1, 2, 3), keepdim=True) + 1e-12),
            g2 / (g2.abs().amax((1, 2, 3), keepdim=True) + 1e-12),
            _corr(r, g1, 5), _corr(r, g2, 5), _corr(r, g2, 9),
            _box(g2.abs().sum(1, keepdim=True), 5) / (g2.abs().sum(1).amax((1, 2)).view(-1, 1, 1, 1) + 1e-12),
            _box((r * g2.sign()).sum(1, keepdim=True), 5) * 20,
            sat,
        ], 1).clamp(-1e3, 1e3)
    assert f.shape[1] == NCH
    return f, lg.detach()


def cache_split(bb, dev, imgs_u8, path, bs):
    if os.path.exists(path) and os.path.exists(path + '.logits.npy'):
        return np.load(path, mmap_mode='r'), np.load(path + '.logits.npy')
    tmp = path + '.tmp.npy'
    N = len(imgs_u8)
    out = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float16, shape=(N, NCH, 64, 64))
    logits = np.zeros((N, 200), np.float32)
    t = time.time()
    for s in range(0, N, bs):
        x = torch.from_numpy(imgs_u8[s:s + bs].transpose(0, 3, 1, 2).astype(np.float32) / 255.).to(dev)
        f, lg = build_features(bb, x)
        out[s:s + bs] = f.cpu().numpy().astype(np.float16)
        logits[s:s + bs] = lg.cpu().numpy()
        if (s // bs) % 20 == 0:
            print(f'   cechy {os.path.basename(path)}: {min(s + bs, N)}/{N}  {time.time() - t:.0f}s', flush=True)
    out.flush(); del out
    np.save(path + '.logits.npy', logits)
    os.replace(tmp, path)
    return np.load(path, mmap_mode='r'), logits


# =============================================================== U-Net
def blk(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU())


class PatchUNet(nn.Module):
    def __init__(self, cin=NCH, w=48):
        super().__init__()
        self.e1, self.e2 = blk(cin, w), blk(w, 2 * w)
        self.e3, self.e4 = blk(2 * w, 4 * w), blk(4 * w, 4 * w)
        self.d3, self.d2, self.d1 = blk(8 * w, 2 * w), blk(4 * w, w), blk(2 * w, w)
        self.out = nn.Conv2d(w, 3, 1)
        self.head = nn.Sequential(nn.Linear(4 * w, 128), nn.SiLU(), nn.Linear(128, 1))

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
        return self.out(d1), g


# =============================================================== symetrie D4
def d4_np(a, k):
    if k & 4: a = np.swapaxes(a, -1, -2)
    if k & 1: a = a[..., :, ::-1]
    if k & 2: a = a[..., ::-1, :]
    return a


def d4_t(t, k):
    if k & 4: t = t.transpose(-1, -2)
    if k & 1: t = t.flip(-1)
    if k & 2: t = t.flip(-2)
    return t


def d4_t_inv(t, k):
    if k & 2: t = t.flip(-2)
    if k & 1: t = t.flip(-1)
    if k & 4: t = t.transpose(-1, -2)
    return t


# =============================================================== trening
def soft_dice(prob, target):
    inter = (prob * target).sum((1, 2))
    den = prob.sum((1, 2)) + target.sum((1, 2))
    has = (target.sum((1, 2)) > 0).float()
    d = 1 - (2 * inter + 1) / (den + 1)
    return (d * has).sum() / has.sum().clamp_min(1)


def train_net(seed, a, dev, feats, feats_aug, aug_ok, rows, tr_idx, mu, sd, out_path):
    print(f'\n--- Trening modelu seed={seed} ---', flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = PatchUNet(w=a.width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = max(1, len(tr_idx) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps)
    wcls = torch.tensor([0.25, 1.0, 0.6], device=dev)
    masks = np.stack([label_mask(r) for r in rows]).astype(np.uint8)
    has = np.array([r['box'] != NONE for r in rows], np.float32)
    mu_t, sd_t = torch.from_numpy(mu).to(dev), torch.from_numpy(sd).to(dev)
    rng = np.random.default_rng(seed)
    for ep in range(a.epochs):
        net.train(); perm = rng.permutation(tr_idx); tot = n = 0; t = time.time()
        for s in range(steps):
            b = np.sort(perm[s * a.bs:(s + 1) * a.bs])
            use_aug = (rng.random(len(b)) < a.p_aug) & aug_ok[b]
            X = np.where(use_aug[:, None, None, None], feats_aug[b], feats[b]).astype(np.float32)
            M = masks[b].copy()
            for j in range(len(b)):
                k = int(rng.integers(8))
                X[j] = d4_np(X[j], k).copy(); M[j] = d4_np(M[j], k).copy()
            x = (torch.from_numpy(np.ascontiguousarray(X)).to(dev) - mu_t) / sd_t
            m = torch.from_numpy(np.ascontiguousarray(M).astype(np.int64)).to(dev)
            h = torch.from_numpy(has[b]).to(dev)
            logit, g = net(x)
            loss = F.cross_entropy(logit, m, weight=wcls) \
                + 0.5 * soft_dice(logit.softmax(1)[:, 1], (m == 1).float()) \
                + 0.3 * F.binary_cross_entropy_with_logits(g, h)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(b); n += len(b)
        print(f'   epoka {ep + 1:3d}/{a.epochs}  strata {tot / n:.4f}  {time.time() - t:.0f}s', flush=True)
    torch.save(net.state_dict(), out_path)
    print(f'   zapisano: {out_path}', flush=True)
    return net


@torch.no_grad()
def infer_one(net, dev, feats, idx, mu, sd, bs=64):
    net.eval()
    mu_t, sd_t = torch.from_numpy(mu).to(dev), torch.from_numpy(sd).to(dev)
    P = np.zeros((len(idx), 3, 64, 64), np.float32)
    G = np.zeros(len(idx), np.float32)
    for s in range(0, len(idx), bs):
        b = np.asarray(idx[s:s + bs])
        order = np.argsort(b)
        x = torch.from_numpy(np.asarray(feats[b[order]], np.float32)).to(dev)
        x = (x - mu_t) / sd_t
        acc = 0; gacc = 0
        for k in range(8):
            lg, g = net(d4_t(x, k))
            acc = acc + d4_t_inv(lg.softmax(1), k); gacc = gacc + g
        p = (acc / 8).cpu().numpy(); gg = (gacc / 8).cpu().numpy()
        P[s + order] = p; G[s + order] = gg
    return P, G


# =============================================================== ramki i decyzja
SHAPES = [(w, h) for w in range(10, 40) for h in range(10, 40)
          if 0.07 <= w * h / 4096 <= 0.165 and 0.45 <= w / h <= 2.2]


def _integral(a):
    return np.pad(np.cumsum(np.cumsum(a, 0), 1), ((1, 0), (1, 0)))


def best_rect(p, A=None):
    I = _integral(p.astype(np.float64))
    A = float(p.sum()) if A is None else float(A)
    best, bb = -1.0, (0, 0, 20, 20)
    for (w, h) in SHAPES:
        s = I[h:, w:] - I[:-h, w:] - I[h:, :-w] + I[:-h, :-w]
        v = s / (w * h + A - s + 1e-9)
        j = int(np.argmax(v))
        if v.ravel()[j] > best:
            best = float(v.ravel()[j])
            yy, xx = np.unravel_index(j, s.shape)
            bb = (int(xx), int(yy), int(xx + w), int(yy + h))
    return bb, best


def _iou(a, b):
    return punkty(a, b) if a != NONE and b != NONE else 0.0


def candidates(P, k_extra=3):
    p1, p12 = P[1], P[1] + P[2]
    out = [best_rect(p1)[0]]
    work = p12.copy()
    for _ in range(k_extra):
        if work.max() < 0.3:
            break
        y, x = np.unravel_index(np.argmax(work), work.shape)
        win = np.zeros_like(work)
        win[max(0, y - 22):y + 23, max(0, x - 22):x + 23] = 1
        b, _ = best_rect(work * win)
        if all(_iou(b, o) < 0.4 for o in out):
            out.append(b)
        work[max(0, b[1] - 2):b[3] + 2, max(0, b[0] - 2):b[2] + 2] = 0
    return out


@torch.no_grad()
def causal(bb, dev, u8_list, cand_list, bs=256):
    var, own = [], []
    for n, (u8, cands) in enumerate(zip(u8_list, cand_list)):
        x = u8.astype(np.float32) / 255.
        t = torch.from_numpy(x.transpose(2, 0, 1))[None]
        clean = _median3(t)[0].numpy().transpose(1, 2, 0)
        var += [x, clean]; own += [(n, -2), (n, -1)]
        for j, b in enumerate(cands):
            v = x.copy(); v[b[1]:b[3], b[0]:b[2]] = clean[b[1]:b[3], b[0]:b[2]]
            var.append(v); own.append((n, j))
    L = []
    for s in range(0, len(var), bs):
        X = torch.from_numpy(np.stack(var[s:s + bs]).transpose(0, 3, 1, 2).copy()).float().to(dev)
        L.append(bb(X).cpu().numpy())
    L = np.concatenate(L)
    res = [dict(drop=np.zeros(len(c)), flip=np.zeros(len(c))) for c in cand_list]
    base = {}
    for (n, j), lg in zip(own, L):
        if j == -2:
            base[n] = [int(lg.argmax()), float(lg.max())]
        elif j == -1:
            c0 = base[n][0]
            base[n] += [base[n][1] - float(lg[c0]), float(lg.argmax() != c0)]
        else:
            c0, l0, dall, _ = base[n]
            res[n]['drop'][j] = (l0 - float(lg[c0])) / (abs(dall) + 0.5)
            res[n]['flip'][j] = float(lg.argmax() != c0)
    for n in range(len(cand_list)):
        res[n]['dall'], res[n]['flip_all'] = base[n][2], base[n][3]
    return res


def describe(P, G, cands, cz):
    p1, p2 = P[1], P[2]
    A = float(p1.sum())
    I1, I2 = _integral(p1.astype(np.float64)), _integral(p2.astype(np.float64))
    C = []
    for j, b in enumerate(cands):
        a = (b[2] - b[0]) * (b[3] - b[1])
        s1 = I1[b[3], b[2]] - I1[b[1], b[2]] - I1[b[3], b[0]] + I1[b[1], b[0]]
        s2 = I2[b[3], b[2]] - I2[b[1], b[2]] - I2[b[3], b[0]] + I2[b[1], b[0]]
        C.append([s1 / a, s2 / a, s1 / (a + A - s1 + 1e-9), cz['drop'][j], cz['flip'][j]])
    return np.array(C, np.float64), dict(g=float(G), A=A / 600., dall=cz['dall'], fa=cz['flip_all'])


def choose(C, prm):
    s = C[:, 0] - prm['gam'] * C[:, 1] + prm['a'] * np.tanh(C[:, 3]) + prm['b'] * C[:, 4]
    return int(np.argmax(s))


def has_feats(C, j, im):
    return np.array([im['g'], C[j, 0], C[j, 2], np.tanh(C[j, 3]), C[j, 4],
                     C[:, 4].max(), im['A'], np.tanh(im['dall']), im['fa'], 1.0])


def fit_logreg(X, y, l2=1e-2, it=3000, lr=0.5):
    mu, sd = X[:, :-1].mean(0), X[:, :-1].std(0) + 1e-6
    Z = np.concatenate([(X[:, :-1] - mu) / sd, X[:, -1:]], 1)
    w = np.zeros(Z.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-Z @ w))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * w)
    return dict(w=w, mu=mu, sd=sd)


def logreg_p(m, X):
    Z = np.concatenate([(X[:, :-1] - m['mu']) / m['sd'], X[:, -1:]], 1)
    return 1 / (1 + np.exp(-Z @ m['w']))


def tune(items):
    grid = [dict(gam=g, a=a, b=b) for g in (0.0, 0.5, 1.0) for a in (0.0, 0.2, 0.5, 1.0)
            for b in (0.0, 0.2, 0.5)]

    def run(tr, te):
        best = None
        for prm in grid:
            iou = np.mean([punkty(gt, cands[choose(C, prm)]) for C, im, cands, gt in tr if gt != NONE])
            if best is None or iou > best[0]:
                best = (iou, prm)
        prm = best[1]
        X = np.array([has_feats(C, choose(C, prm), im) for C, im, _, _ in tr])
        y = np.array([gt != NONE for _, _, _, gt in tr], np.float64)
        lr = fit_logreg(X, y)

        def score(items_, kappa):
            return score_items(items_, dict(prm=prm, lr=lr, kappa=kappa))
        kappa = max(np.arange(0.4, 1.81, 0.1), key=lambda k: score(tr, k))
        return prm, lr, float(kappa), score(te, kappa)

    rng = np.random.default_rng(0); perm = rng.permutation(len(items)); h = len(items) // 2
    A = [items[i] for i in perm[:h]]; B = [items[i] for i in perm[h:]]
    s_ab, s_ba = run(A, B)[3], run(B, A)[3]
    prm, lr, kappa, _ = run(items, items)
    return dict(prm=prm, lr=lr, kappa=kappa, cv=(s_ab + s_ba) / 2)


def answer(C, im, cands, model):
    j = choose(C, model['prm'])
    p = logreg_p(model['lr'], has_feats(C, j, im)[None])[0]
    e = np.clip(model['kappa'] * C[j, 2], 0.05, 1.0)
    return cands[j] if p * e > 1 - p else NONE


def score_items(its, model):
    return float(np.mean([punkty(gt, answer(C, im, cands, model)) for C, im, cands, gt in its]))


def decide_all(Ps, Gs, cand_list, cz_list, model):
    out = []
    for P, G, cands, cz in zip(Ps, Gs, cand_list, cz_list):
        C, im = describe(P, G, cands, cz)
        out.append(answer(C, im, cands, model))
    return out


# =============================================================== główny przebieg
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='submission_ensemble.csv')
    ap.add_argument('--cache', default='cache_ens')
    ap.add_argument('--models_dir', default='models_ensemble',
                    help='folder gdzie zapisywane sa unet_1.pt ... unet_4.pt')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--width', type=int, default=0, help='0 = auto (64 na GPU, 32 na CPU)')
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--p_aug', type=float, default=0.35)
    ap.add_argument('--holdout', type=int, default=500)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--extra', default='', help='folder z generate.py (np. synth_final)')
    ap.add_argument('--seeds', default='42,123,777,2026',
                    help='ziarna losowe modeli, po przecinku')
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(',')]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if a.width == 0:
        a.width = 64 if dev != 'cpu' else 32
    if a.smoke:
        a.epochs, a.width, a.holdout = 1, 16, 64
        a.out = 'SMOKE_' + a.out
        seeds = seeds[:2]   # tylko 2 modele przy smoke
    os.makedirs(a.cache, exist_ok=True)
    os.makedirs(a.models_dir, exist_ok=True)
    print(f'urzadzenie: {dev}, szerokosc: {a.width}, epoki: {a.epochs}, modele: {seeds}')

    # ---- dane
    rows = read_train(a.data)
    test_ids = read_test(a.data)
    if a.smoke:
        rows, test_ids = rows[:320], test_ids[:64]
    n_real = len(rows)
    if a.extra:
        ex = read_train(a.extra)
        rows += ex[:200] if a.smoke else ex
        print(f'dodatkowe obrazki (synth): {len(rows) - n_real}')
    bb = load_backbone(a.data, dev)

    # ---- cechy (liczone raz, wspólne dla wszystkich modeli)
    print('\n[1/4] Obliczanie cech (wspólne dla wszystkich modeli)')
    tr_u8 = np.stack([load_u8(r['root'], 'train', r['id']) for r in rows])
    rng0 = np.random.default_rng(123)
    qs = rng0.integers(70, 89, len(rows))
    p_aug_u8 = os.path.join(a.cache, 'train_aug_u8.npy')
    if os.path.exists(p_aug_u8):
        aug_u8 = np.load(p_aug_u8)
    else:
        aug_u8 = np.stack([rejpeg(u, q) for u, q in zip(tr_u8, qs)])
        np.save(p_aug_u8, aug_u8)
    te_u8 = np.stack([load_u8(a.data, 'test', i) for i in test_ids])
    fb = 128 if dev != 'cpu' else 32
    f_tr, lg_tr = cache_split(bb, dev, tr_u8, os.path.join(a.cache, 'f_train.npy'), fb)
    f_au, lg_au = cache_split(bb, dev, aug_u8, os.path.join(a.cache, 'f_train_jpeg.npy'), fb)
    f_te, _ = cache_split(bb, dev, te_u8, os.path.join(a.cache, 'f_test.npy'), fb)
    aug_ok = np.array([r['box'] == NONE or (r['true'] >= 0 and int(l.argmax()) != r['true'])
                       for r, l in zip(rows, lg_au)])

    # ---- podział train/val (ten sam dla wszystkich modeli)
    perm = np.random.default_rng(0).permutation(n_real)
    va_idx = np.sort(perm[:a.holdout])
    tr_idx = np.sort(np.concatenate([perm[a.holdout:], np.arange(n_real, len(rows))]))
    smp = np.asarray(f_tr[tr_idx[:min(400, len(tr_idx))]], np.float32)
    mu = smp.mean((0, 2, 3), dtype=np.float64).reshape(1, -1, 1, 1).astype(np.float32)
    sd = (smp.std((0, 2, 3), dtype=np.float64) + 1e-4).reshape(1, -1, 1, 1).astype(np.float32)
    del smp

    # ---- trenowanie 4 modeli
    print(f'\n[2/4] Trenowanie {len(seeds)} modeli U-Net')
    nets = []
    for i, seed in enumerate(seeds):
        out_path = os.path.join(a.models_dir, f'unet_{i + 1}.pt')
        if os.path.exists(out_path):
            print(f'   Model {i + 1} (seed={seed}): wczytywanie z {out_path}')
            net = PatchUNet(w=a.width).to(dev)
            net.load_state_dict(torch.load(out_path, map_location=dev, weights_only=True))
            nets.append(net)
        else:
            net = train_net(seed, a, dev, f_tr, f_au, aug_ok, rows, tr_idx, mu, sd, out_path)
            nets.append(net)

    # ---- walidacja (ensemble na danych val)
    print('\n[3/4] Walidacja ensemble na odlozonych obrazkach')

    def infer_ensemble(feats, idx):
        """Uśrednienie predykcji ze wszystkich modeli."""
        P_sum = np.zeros((len(idx), 3, 64, 64), np.float32)
        G_sum = np.zeros(len(idx), np.float32)
        for net in nets:
            P, G = infer_one(net, dev, feats, idx, mu, sd)
            P_sum += P; G_sum += G
        return P_sum / len(nets), G_sum / len(nets)

    items = []
    for name, feats, u8s in (('oryginal', f_tr, tr_u8), ('po JPEG q70-88', f_au, aug_u8)):
        ok = va_idx if name == 'oryginal' else va_idx[aug_ok[va_idx]]
        P, G = infer_ensemble(feats, ok)
        cl = [candidates(p) for p in P]
        cz = causal(bb, dev, [u8s[i] for i in ok], cl)
        for n, i in enumerate(ok):
            C, im = describe(P[n], G[n], cl[n], cz[n])
            items.append((C, im, cl[n], rows[i]['box']))
    model = tune(items)
    print(f'   wynik walidacji krzyzowej (ensemble): {model["cv"]:.4f}')
    print(f'   parametry: {model["prm"]}, kappa={model["kappa"]:.1f}')

    # ---- test (ensemble)
    print('\n[4/4] Inferencja na zbiorze testowym (ensemble)')
    ans = []
    for s in range(0, len(test_ids), 500):
        idx = np.arange(s, min(s + 500, len(test_ids)))
        P, G = infer_ensemble(f_te, idx)
        cl = [candidates(p) for p in P]
        cz = causal(bb, dev, [te_u8[i] for i in idx], cl)
        ans += decide_all(P, G, cl, cz, model)
        print(f'   {idx[-1] + 1}/{len(test_ids)}', flush=True)

    with open(a.out, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['id', 'x_min', 'y_min', 'x_max', 'y_max'])
        for iid, b in zip(test_ids, ans):
            assert b == NONE or (0 <= b[0] < b[2] <= 64 and 0 <= b[1] < b[3] <= 64), b
            w.writerow([iid, *[int(v) for v in b]])
    nb = sum(b != NONE for b in ans)
    print(f'\nGotowe! Zapisano {a.out}: {len(ans)} wierszy, ramek {nb} ({100 * nb / len(ans):.1f}%)')
    print(f'Modele zapisane w: {a.models_dir}/')


if __name__ == '__main__':
    main()
