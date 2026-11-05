"""
Etap 4: LightGBM lambdarank / regresja IoU na cechach z rank_cache.py.
Walidacja krzyzowa PO OBRAZKACH; ocena na 500 val (prawdziwa vs synth osobno).

    python exp/rank_train.py
"""
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import punkty, NONE

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


def load_split(path):
    z = np.load(path, allow_pickle=True)
    F = list(z['F']); R = list(z['R']); y = list(z['y']); gid = z['gid']
    return F, R, y, gid


def flatten(F, y, gid):
    Xs, ys, gs = [], [], []
    for i, (f, yy) in enumerate(zip(F, y)):
        Xs.append(f); ys.append(yy); gs.append(np.full(len(yy), gid[i]))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def group_sizes(gs):
    # lightgbm wymaga liczby wierszy w kazdej grupie w kolejnosci
    _, idx, counts = np.unique(gs, return_index=True, return_counts=True)
    order = np.argsort(idx)
    return counts[order]


def cv_rank(F, R, y, gid, folds=5, seed=0, objective='lambdarank'):
    ug = np.unique(gid)
    fold = dict(zip(ug, np.random.default_rng(seed).integers(0, folds, len(ug))))
    pred_box = {}
    for k in range(folds):
        tr = [i for i in range(len(F)) if fold[gid[i]] != k]
        te = [i for i in range(len(F)) if fold[gid[i]] == k]
        Xtr, ytr, gtr = flatten([F[i] for i in tr], [y[i] for i in tr], [gid[i] for i in tr])
        # etykiety rankingowe: dyskretyzacja IoU
        if objective == 'lambdarank':
            y_lab = np.clip((ytr * 10).astype(int), 0, 10)
            ds = lgb.Dataset(Xtr, y_lab, group=group_sizes(gtr))
            params = dict(objective='lambdarank', metric='ndcg', learning_rate=0.05,
                          num_leaves=31, min_data_in_leaf=20, verbosity=-1, seed=0)
            bst = lgb.train(params, ds, num_boost_round=200)
        else:
            ds = lgb.Dataset(Xtr, ytr, group=group_sizes(gtr))
            params = dict(objective='regression', metric='l2', learning_rate=0.05,
                          num_leaves=31, min_data_in_leaf=20, verbosity=-1, seed=0)
            bst = lgb.train(params, ds, num_boost_round=200)
        for i in te:
            sc = bst.predict(F[i])
            pred_box[i] = R[i][int(np.argmax(sc))]
            pred_iou = y[i][int(np.argmax(sc))]
            pred_box[i] = (R[i][int(np.argmax(sc))], float(pred_iou), float(sc.max()))
    return pred_box, bst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='cache_rank/train_feats.npz')
    ap.add_argument('--val', default='cache_rank/val_feats.npz')
    ap.add_argument('--out', default='cache_rank/ranker.txt')
    a = ap.parse_args()
    assert lgb is not None

    Ft, Rt, yt, gt = load_split(a.train)
    print(f'trening: {len(Ft)} obrazkow, srednio {np.mean([len(f) for f in Ft]):.0f} wariantow')

    for obj in ('lambdarank', 'regression'):
        pred, _ = cv_rank(Ft, Rt, yt, gt, objective=obj)
        ious = [pred[i][1] for i in range(len(Ft))]
        print(f'CV {obj:12s} IoU ramki {np.mean(ious):.3f}  (wyrocznia {np.mean([yy.max() for yy in yt]):.3f})')

    # finalny model na calosci, ocena na val
    Xtr, ytr, gtr = flatten(Ft, yt, gt)
    y_lab = np.clip((ytr * 10).astype(int), 0, 10)
    ds = lgb.Dataset(Xtr, y_lab, group=group_sizes(gtr))
    params = dict(objective='lambdarank', metric='ndcg', learning_rate=0.05,
                  num_leaves=31, min_data_in_leaf=20, verbosity=-1, seed=0)
    bst = lgb.train(params, ds, num_boost_round=300)
    bst.save_model(a.out)

    if os.path.exists(a.val):
        Fv, Rv, yv, gv = load_split(a.val)
        ious = []
        for i in range(len(Fv)):
            sc = bst.predict(Fv[i])
            j = int(np.argmax(sc))
            ious.append(yv[i][j])
        print(f'val lambdarank IoU ramki {np.mean(ious):.3f}  wyrocznia {np.mean([yy.max() for yy in yv]):.3f}')
        # baseline: wariant startowy (flaga start w cechach — kolumna z float(r==r0); u nas index z unet_mass)
        # przyblizenie: argmax cechy oczekiwane IoU na mapie (przedostatnie przed jpeg)
        # tu: pierwszy wariant w liscie = start z top_starts
        base = [yv[i][0] for i in range(len(Fv))]
        print(f'val start (wariant 0) IoU {np.mean(base):.3f}')


if __name__ == '__main__':
    main()
