"""
P1 v2: ensemble U-Netow (modele_natrenowane/) + mapa przyczynowa (exp/causal_cache.py).

    python exp/causal_cache.py val && python exp/causal_cache.py test     # ~45 min na RTX 2060
    python solve_v2.py --out sub_v2.csv

Ramka: maksymalizacja oczekiwanego IoU na mapie P(latka) z U-Neta pomnozonej przez
znormalizowana mape przyczynowa (spadek marginesu klasy modelu, gdy okno zastapimy
wersja po mocnym JPEG). Latka ma minimalne eps, wiec jej usuniecie przywraca klase;
usuniecie wabika nie zmienia nic. Decyzja ramka / 0,0,0,0: regresja logistyczna
na cechach obrazka, prog strojony walidacja krzyzowa pod funkcje oceny.
"""
import argparse, csv, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp'))
import ensemble4 as E
from common import punkty, NONE, train_rows, test_ids, load_imgs, best_rect, integral

D = 'data/p1_student'


def unet_maps(dev, imgs, nets, mu, sd, bb):
    f = []
    for s in range(0, len(imgs), 64):
        x = torch.from_numpy(imgs[s:s + 64].transpose(0, 3, 1, 2).astype(np.float32) / 255).to(dev)
        f.append(E.build_features(bb, x)[0].cpu().numpy().astype(np.float16))
    f = np.concatenate(f)
    P = 0; G = 0
    for n in nets:
        p, g = E.infer_one(n, dev, f, np.arange(len(f)), mu, sd)
        P = P + p / len(nets); G = G + g / len(nets)
    return P.astype(np.float16), G


def cached_unet(split, dev, imgs, get_ctx, tags=('unet',)):
    """Srednia map z kilku zestawow U-Netow (cache_v2/{tag}_{split}.npz); 'unet' = modele_natrenowane/."""
    P = G = 0
    for tag in tags:
        p = f'cache_v2/{tag}_{split}.npz'
        if not os.path.exists(p):
            assert tag == 'unet', f'brak {p} (uruchom train_v3.py)'
            Pu, Gu = unet_maps(dev, imgs(), *get_ctx())
            np.savez_compressed(p, P=Pu, G=Gu)
        z = np.load(p)
        P = P + z['P'].astype(np.float32) / len(tags); G = G + z['G'] / len(tags)
    return P, G


def box_sum(I, b):
    return I[b[3], b[2]] - I[b[1], b[2]] - I[b[3], b[0]] + I[b[1], b[0]]


def analyse(P, G, C, st, a):
    """-> ramki, cechy obrazka (do decyzji)."""
    boxes, feats = [], []
    for p, g, c, s in zip(P, G, C, st):
        cz = np.clip(c.mean(0), 0, None); cz = cz / (cz.max() + 1e-9)
        pc = p[1] * cz ** a
        b, v = best_rect(pc)
        ar = (b[2] - b[0]) * (b[3] - b[1])
        I1, I2, Ic = (integral(m.astype(np.float64)) for m in (p[1], p[2], cz))
        m0, mf, dmax = s[:, 0].mean(), s[:, 1].mean(), s[:, 2].max()
        feats.append([g, p[1].sum() / 600, p[2].sum() / 600, box_sum(I1, b) / ar, box_sum(I2, b) / ar,
                      box_sum(Ic, b) / ar, v, pc.sum() / 600, np.tanh(m0 / 5), np.tanh(mf / 5),
                      float((s[:, 1] < 0).mean()), np.tanh(dmax), 1.0])
        boxes.append(b)
    return boxes, np.array(feats, np.float64)


def fit_and_score(Ftr, gt_tr, boxes_tr, Fte, boxes_te, gt_te=None):
    y = np.array([g != NONE for g in gt_tr], np.float64)
    lr = E.fit_logreg(Ftr, y)
    iou = np.array([punkty(g, b) for g, b in zip(gt_tr, boxes_tr)])
    # oczekiwane IoU: liniowo z cech ramki, uczone na obrazkach z latka
    m = y == 1
    Xe = Ftr[m][:, [3, 5, 6, 12]]
    we = np.linalg.lstsq(Xe, iou[m], rcond=None)[0]

    def decide(F, boxes, t):
        p = E.logreg_p(lr, F)
        e = np.clip(F[:, [3, 5, 6, 12]] @ we, 0.05, 1.0)
        return [b if pi * e_ - (1 - pi) > t else NONE for b, pi, e_ in zip(boxes, p, e)]

    ts = np.linspace(-0.5, 0.5, 41)
    sc = [np.mean([punkty(g, o) for g, o in zip(gt_tr, decide(Ftr, boxes_tr, t))]) for t in ts]
    t = float(ts[int(np.argmax(sc))])
    out = decide(Fte, boxes_te, t)
    return out, t, (None if gt_te is None else np.mean([punkty(g, o) for g, o in zip(gt_te, out)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='sub_v2.csv')
    ap.add_argument('--alpha', type=float, default=1.0, help='wykladnik mapy przyczynowej (0 = sam U-Net)')
    ap.add_argument('--unet', default='unet', help="zestawy map U-Neta do usrednienia, np. 'unet,unet3'")
    ap.add_argument('--refine', action='store_true', help='dopracuj krawedzie ramek testem przyczynowym (~15 min)')
    ap.add_argument('--mbr', action='store_true',
                    help='z --refine: wybierz wariant max. oczekiwanego IoU (softmax po wyniku/T)')
    a = ap.parse_args()
    tags = a.unet.split(',')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('cache_v2', exist_ok=True)
    rows = train_rows()
    perm = np.random.default_rng(0).permutation(len(rows))
    va = np.sort(perm[:500])
    ctx = {}

    def get_ctx():
        if not ctx:
            bb = E.load_backbone(D, dev)
            trs = load_imgs([rows[i]['id'] for i in np.sort(perm[500:])[:400]], 'train')
            f = []
            for s in range(0, len(trs), 64):
                x = torch.from_numpy(trs[s:s + 64].transpose(0, 3, 1, 2).astype(np.float32) / 255).to(dev)
                f.append(E.build_features(bb, x)[0].cpu().numpy())
            smp = np.concatenate(f).astype(np.float16).astype(np.float32)
            mu = smp.mean((0, 2, 3), dtype=np.float64).reshape(1, -1, 1, 1).astype(np.float32)
            sd = (smp.std((0, 2, 3), dtype=np.float64) + 1e-4).reshape(1, -1, 1, 1).astype(np.float32)
            nets = []
            for k in range(1, 7):
                n = E.PatchUNet(w=96).to(dev)
                n.load_state_dict(torch.load(f'modele_natrenowane/unet_{k}.pt', map_location=dev, weights_only=True))
                nets.append(n)
            ctx['v'] = (nets, mu, sd, bb)
        return ctx['v']

    Pv, Gv = cached_unet('val', dev, lambda: load_imgs([rows[i]['id'] for i in va], 'train'), get_ctx, tags)
    Cv = np.load('cache_v2/causal_val.npy').astype(np.float32); Sv = np.load('cache_v2/causal_val_st.npy')
    gt = [rows[i]['box'] for i in va]
    bv, Fv = analyse(Pv, Gv, Cv, Sv, a.alpha)
    has = np.array([g != NONE for g in gt])
    print(f'walidacja: IoU ramki na obrazkach z latka {np.mean([punkty(g, b) for g, b in zip(gt, bv) if g != NONE]):.3f}')
    # uczciwa ocena: 5-krotna walidacja krzyzowa decyzji
    fold = np.random.default_rng(0).integers(0, 5, len(va)); res = np.zeros(len(va))
    for k in range(5):
        tr, te = np.nonzero(fold != k)[0], np.nonzero(fold == k)[0]
        out, _, _ = fit_and_score(Fv[tr], [gt[i] for i in tr], [bv[i] for i in tr], Fv[te], [bv[i] for i in te])
        res[te] = [punkty(gt[i], o) for i, o in zip(te, out)]
    print(f'walidacja krzyzowa (5-fold) wynik: {res.mean():.4f}   (z latka {res[has].mean():.3f}, bez {res[~has].mean():.3f})')

    if not os.path.exists('cache_v2/causal_test.done') or int(open('cache_v2/causal_test.done').read()) < 5000:
        print('brak pelnych map przyczynowych testu - koniec'); return
    ids = test_ids()
    Pt, Gt = cached_unet('test', dev, lambda: load_imgs(ids, 'test'), get_ctx, tags)
    Ct = np.load('cache_v2/causal_test.npy').astype(np.float32); St = np.load('cache_v2/causal_test_st.npy')
    bt, Ft = analyse(Pt, Gt, Ct, St, a.alpha)
    out, t, _ = fit_and_score(Fv, gt, bv, Ft, bt)
    if a.refine:
        import refine_v2 as R
        bb = E.load_backbone(D, dev)
        adv = np.nonzero(has)[0]
        Uv = load_imgs([rows[va[i]]['id'] for i in adv], 'train')
        Tv = R.table(bb, Uv, Pv[adv, 1], Cv[adv], dev)
        prm = R.tune(Tv, [gt[i] for i in adv])
        temp = R.tune_mbr(Tv, [gt[i] for i in adv], *prm) if a.mbr else None
        def pick(r, F):
            return r[R.choose_mbr(r, F, *prm, temp)] if a.mbr else r[R.choose(F, *prm)]
        # uczciwa ocena refine: 5-fold CV (takze T dla MBR)
        fold = np.random.default_rng(1).integers(0, 5, len(adv)); cv_iou = np.zeros(len(adv))
        for k in range(5):
            tr, te = np.nonzero(fold != k)[0], np.nonzero(fold == k)[0]
            p = R.tune([Tv[i] for i in tr], [gt[adv[i]] for i in tr])
            t_ = R.tune_mbr([Tv[i] for i in tr], [gt[adv[i]] for i in tr], *p) if a.mbr else None
            for i in te:
                r, F = Tv[i]
                cv_iou[i] = punkty(gt[adv[i]], r[R.choose_mbr(r, F, *p, t_)] if a.mbr else r[R.choose(F, *p)])
        print(f'dopracowanie krawedzi{"+MBR" if a.mbr else ""}: IoU {np.mean([punkty(gt[i], bv[i]) for i in adv]):.3f} -> '
              f'{cv_iou.mean():.3f} (5-fold CV; parametry {prm}' + (f', T={temp}' if a.mbr else '') + ')')
        sel = [i for i, o in enumerate(out) if o != NONE]
        Ut = load_imgs([ids[i] for i in sel], 'test')
        Tt = R.table(bb, Ut, Pt[sel, 1], Ct[sel], dev)
        for i, (r, F) in zip(sel, Tt):
            out[i] = pick(r, F)
    with open(a.out, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['id', 'x_min', 'y_min', 'x_max', 'y_max'])
        for iid, b in zip(ids, out):
            assert b == NONE or (0 <= b[0] < b[2] <= 64 and 0 <= b[1] < b[3] <= 64), b
            w.writerow([iid, *[int(v) for v in b]])
    print(f'zapisano {a.out}: ramek {np.mean([o != NONE for o in out]) * 100:.1f}%  (prog {t:+.2f})')


if __name__ == '__main__':
    main()
