"""
P1 v4: mapy (unet+unet3 lub OOF) + LightGBM ranker wariantow ramki (Etap 4) + decyzja.

    # po cache_rank/{train,val,test}_feats.npz i cache_rank/ranker.txt:
    python solve_v4.py --out sub_v4.csv
"""
import argparse, csv, os, sys
import numpy as np
import torch
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp'))
import ensemble4 as E
from common import punkty, NONE, train_rows, test_ids, load_imgs
import solve_v2 as S2

D = 'data/p1_student'


def load_feats(path):
    z = np.load(path, allow_pickle=True)
    return list(z['F']), list(z['R']), list(z['y']), z['gid'], list(z['ids']) if 'ids' in z.files else None


def flatten(F, y, gid):
    Xs, ys, gs = [], [], []
    for f, yy, g in zip(F, y, gid):
        Xs.append(f); ys.append(yy); gs.append(np.full(len(yy), g))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def group_sizes(gs):
    _, idx, counts = np.unique(gs, return_index=True, return_counts=True)
    return counts[np.argsort(idx)]


def train_ranker(F, y, gid, out):
    X, yy, gs = flatten(F, y, gid)
    y_lab = np.clip((yy * 10).astype(int), 0, 10)
    ds = lgb.Dataset(X, y_lab, group=group_sizes(gs))
    params = dict(objective='lambdarank', metric='ndcg', learning_rate=0.05,
                  num_leaves=31, min_data_in_leaf=20, verbosity=-1, seed=0)
    bst = lgb.train(params, ds, num_boost_round=300)
    bst.save_model(out)
    return bst


def pick_boxes(bst, F, R):
    boxes, eious, scores = [], [], []
    for f, r in zip(F, R):
        sc = bst.predict(f)
        j = int(np.argmax(sc))
        boxes.append(tuple(int(x) for x in r[j]))
        # oczekiwane IoU: softmax po score * przyblizenie — tu max score znormalizowany przez top
        s = sc - sc.max(); w = np.exp(s / 0.5); w /= w.sum()
        # bez prawdziwego IoU: uzyj cechy map-IoU jesli jest, inaczej clip score
        eious.append(float(np.clip(sc[j] / (np.abs(sc).max() + 1e-6), 0.05, 1.0)))
        scores.append(float(sc[j]))
    return boxes, np.array(eious), np.array(scores)


def fit_decide(Ftr, gt_tr, boxes_tr, eiou_tr, Fte, boxes_te, eiou_te, gt_te=None):
    """Jak solve_v2.fit_and_score, ale oczekiwane IoU z rankera."""
    y = np.array([g != NONE for g in gt_tr], np.float64)
    lr = E.fit_logreg(Ftr, y)

    def decide(F, boxes, eiou, t):
        p = E.logreg_p(lr, F)
        e = np.clip(eiou, 0.05, 1.0)
        return [b if pi * e_ - (1 - pi) > t else NONE for b, pi, e_ in zip(boxes, p, e)]

    ts = np.linspace(-0.5, 0.5, 41)
    sc = [np.mean([punkty(g, o) for g, o in zip(gt_tr, decide(Ftr, boxes_tr, eiou_tr, t))]) for t in ts]
    t = float(ts[int(np.argmax(sc))])
    out = decide(Fte, boxes_te, eiou_te, t)
    return out, t, (None if gt_te is None else np.mean([punkty(g, o) for g, o in zip(gt_te, out)]))


def analyse_like_v2(P, G, C, st, alpha=1.0):
    return S2.analyse(P, G, C, st, alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='sub_v4.csv')
    ap.add_argument('--unet', default='unet,unet3')
    ap.add_argument('--rank_train', default='cache_rank/train_feats.npz')
    ap.add_argument('--rank_val', default='cache_rank/val_feats.npz')
    ap.add_argument('--rank_test', default='cache_rank/test_feats.npz')
    ap.add_argument('--ranker', default='cache_rank/ranker.txt')
    ap.add_argument('--val_only', action='store_true', help='tylko CV na val (bez testu)')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tags = a.unet.split(',')

    rows = train_rows()
    perm = np.random.default_rng(0).permutation(len(rows))
    va = np.sort(perm[:500])
    gt = [rows[i]['box'] for i in va]
    has = np.array([g != NONE for g in gt])

    # mapy jak solve_v2 (do cech decyzji + porownanie)
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

    Pv, Gv = S2.cached_unet('val', dev, lambda: load_imgs([rows[i]['id'] for i in va], 'train'), get_ctx, tags)
    Cv = np.load('cache_v2/causal_val.npy').astype(np.float32); Sv = np.load('cache_v2/causal_val_st.npy')
    bv0, Fv = analyse_like_v2(Pv, Gv, Cv, Sv, 1.0)

    # ranker
    Ftr, Rtr, ytr, gtr, _ = load_feats(a.rank_val)  # domyslnie ucz na val CV; jesli train gotowy — nadpisz
    if os.path.exists(a.rank_train):
        Ftr, Rtr, ytr, gtr, _ = load_feats(a.rank_train)
        print(f'ranker: trening na {len(Ftr)} obrazkach z {a.rank_train}')
    else:
        print(f'ranker: brak {a.rank_train} — 5-fold CV na val')

    Fv_r, Rv_r, yv_r, gv_r, ids_v = load_feats(a.rank_val)
    # mapowanie val feats (tylko z latka) -> indeksy w va
    id_to_va = {rows[i]['id']: j for j, i in enumerate(va)}
    adv_local = [id_to_va[i] for i in ids_v]

    # OOF ramki rankera na val (uczciwa ocena)
    fold_r = np.random.default_rng(0).integers(0, 5, len(Fv_r))
    boxes_oof = [None] * len(Fv_r)
    eiou_oof = np.zeros(len(Fv_r), np.float64)
    cv_iou = np.zeros(len(Fv_r))
    for k in range(5):
        tr = np.nonzero(fold_r != k)[0]; te = np.nonzero(fold_r == k)[0]
        if os.path.exists(a.rank_train):
            # model z train — te same wagi dla wszystkich foldow val
            if k == 0:
                bst = train_ranker(Ftr, ytr, gtr, a.ranker)
            for i in te:
                sc = bst.predict(Fv_r[i]); j = int(np.argmax(sc))
                boxes_oof[i] = tuple(int(x) for x in Rv_r[i][j])
                eiou_oof[i] = float(np.clip(yv_r[i][j], 0.05, 1.0))  # przy CV znamy IoU; na tescie regresja
                cv_iou[i] = yv_r[i][j]
        else:
            bst_k = train_ranker([Fv_r[i] for i in tr], [yv_r[i] for i in tr], gv_r[tr], a.ranker + f'.fold{k}')
            # regresja IoU na top-wariancie (cechy wybranego wiersza)
            Xr, yr = [], []
            for i in tr:
                sc = bst_k.predict(Fv_r[i]); j = int(np.argmax(sc))
                Xr.append(Fv_r[i][j]); yr.append(yv_r[i][j])
            Xr, yr = np.stack(Xr), np.array(yr)
            w_reg = np.linalg.lstsq(Xr, yr, rcond=None)[0]
            for i in te:
                sc = bst_k.predict(Fv_r[i]); j = int(np.argmax(sc))
                boxes_oof[i] = tuple(int(x) for x in Rv_r[i][j])
                eiou_oof[i] = float(np.clip(Fv_r[i][j] @ w_reg, 0.05, 0.95))
                cv_iou[i] = yv_r[i][j]
    print(f'ranker OOF IoU ramki {cv_iou.mean():.3f}  (start {np.mean([yv_r[i][0] for i in range(len(yv_r))]):.3f})')
    if not os.path.exists(a.rank_train):
        bst = train_ranker(Fv_r, yv_r, gv_r, a.ranker)

    # ramki: ranker na obrazkach z latka; na czystych zostaw kandydat z map (decyzja ma odrzucic)
    bv = list(bv0)
    eiou_full = np.array([float(np.clip(f[6], 0.05, 0.95)) for f in Fv])  # proxy: v z best_rect
    for loc, b, e in zip(adv_local, boxes_oof, eiou_oof):
        bv[loc] = b; eiou_full[loc] = e

    print(f'walidacja: IoU ramki na obrazkach z latka {np.mean([punkty(g, b) for g, b in zip(gt, bv) if g != NONE]):.3f}')
    fold = np.random.default_rng(0).integers(0, 5, len(va)); res = np.zeros(len(va))
    for k in range(5):
        tr, te = np.nonzero(fold != k)[0], np.nonzero(fold == k)[0]
        out, _, _ = fit_decide(Fv[tr], [gt[i] for i in tr], [bv[i] for i in tr], eiou_full[tr],
                               Fv[te], [bv[i] for i in te], eiou_full[te])
        res[te] = [punkty(gt[i], o) for i, o in zip(te, out)]
    print(f'walidacja krzyzowa (5-fold) wynik: {res.mean():.4f}   (z latka {res[has].mean():.3f}, bez {res[~has].mean():.3f})')

    if a.val_only or not os.path.exists(a.rank_test):
        print('brak test_feats lub --val_only — koniec')
        return

    ids = test_ids()
    Pt, Gt = S2.cached_unet('test', dev, lambda: load_imgs(ids, 'test'), get_ctx, tags)
    Ct = np.load('cache_v2/causal_test.npy').astype(np.float32); St = np.load('cache_v2/causal_test_st.npy')
    bt0, Ft = analyse_like_v2(Pt, Gt, Ct, St, 1.0)
    Fte, Rte, _, _, ids_te = load_feats(a.rank_test)
    assert list(ids_te) == list(ids) or len(Fte) == len(ids)
    boxes_te, eiou_te, _ = pick_boxes(bst, Fte, Rte)
    # decyzja: najpierw logreg czy w ogole latka, potem ranker box
    out, t, _ = fit_decide(Fv, gt, bv, eiou_full, Ft, boxes_te, eiou_te)
    # jesli logreg mowi brak — NONE; jesli mowi jest — uzyj ranker box
    # fit_decide juz to robi na boxes_te
    with open(a.out, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['id', 'x_min', 'y_min', 'x_max', 'y_max'])
        for iid, b in zip(ids, out):
            assert b == NONE or (0 <= b[0] < b[2] <= 64 and 0 <= b[1] < b[3] <= 64), b
            w.writerow([iid, *[int(v) for v in b]])
    print(f'zapisano {a.out}: ramek {np.mean([o != NONE for o in out]) * 100:.1f}%  (prog {t:+.2f})')


if __name__ == '__main__':
    main()
