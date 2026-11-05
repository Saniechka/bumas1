"""
Etap 2: U-Net K=4 poza-proba (OOF).
Kazdy obrazek treningowy/syntetyczny dostaje mape z modelu, ktory go nie widzial;
walidacja 500 i test — srednia z 4 foldow. Mapki: memmap float16 w cache_oof/.

    python train_oof.py --synth synth_v2,synth_final --synth_max 6000,6000
Wznawia sie: pomija foldi z istniejacym unet_fold{k}.pt i gotowymi mapami.
"""
import argparse, csv, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'exp'))
import ensemble4 as E
from common import punkty, NONE, best_rect, D
from causal_cache import compute_causal, QS
from train_v3 import Stack, feats_cache, train, infer, report, load_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synth', default='synth_v2,synth_final')
    ap.add_argument('--synth_max', default='6000,6000')
    ap.add_argument('--cache', default='cache_v3')
    ap.add_argument('--out', default='cache_oof')
    ap.add_argument('--folds', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    dev = 'cuda'
    os.makedirs(a.cache, exist_ok=True); os.makedirs(a.out, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    bb = E.load_backbone(D, dev)

    real = E.read_train(D)
    perm = np.random.default_rng(0).permutation(len(real))
    va_real = np.sort(perm[:500])
    rows = list(real)
    for k, (d, mx) in enumerate(zip(a.synth.split(','), map(int, a.synth_max.split(',')))):
        if not os.path.exists(os.path.join(d, 'train_labels.csv')):
            print(f'pomijam {d}'); continue
        ex = E.read_train(d)
        ex = [ex[i] for i in np.random.default_rng(k).permutation(len(ex))[:mx]]
        rows += ex
        print(f'{d}: {len(ex)}')
    # fold tylko na indeksach poza va_real (OOF); va_real i test = srednia z foldow
    pool = np.setdiff1d(np.arange(len(rows)), va_real)
    fold_of = np.full(len(rows), -1, int)
    fold_of[pool] = np.random.default_rng(a.seed).integers(0, a.folds, len(pool))

    print('[1/3] cechy i mapy przyczynowe')
    U = np.stack([E.load_u8(r['root'], 'train', r['id']) for r in rows])
    f_tr, _ = feats_cache(bb, dev, U, os.path.join(a.cache, 'f_train.npy'))
    c_tr, _ = compute_causal(bb, U, os.path.join(a.cache, 'causal_train.npy'), dev, 'trening')
    ids = [r['id'] for r in csv.DictReader(open(os.path.join(D, 'test_ids.csv')))]
    Ut = load_set(D, 'test', ids)
    f_te, _ = feats_cache(bb, dev, Ut, os.path.join(a.cache, 'f_test.npy'))
    c_te, _ = compute_causal(bb, Ut, 'cache_v2/causal_test.npy', dev, 'test')
    del U, Ut
    X, Xt = Stack(f_tr, c_tr), Stack(f_te, c_te)

    P_all = np.lib.format.open_memmap(os.path.join(a.out, 'P_train.f16.npy'), 'w+', np.float16,
                                      (len(rows), 3, 64, 64))
    G_all = np.lib.format.open_memmap(os.path.join(a.out, 'G_train.f32.npy'), 'w+', np.float32, (len(rows),))
    P_te = np.zeros((len(Xt), 3, 64, 64), np.float32)
    G_te = np.zeros(len(Xt), np.float32)
    P_va = np.zeros((len(va_real), 3, 64, 64), np.float32)
    G_va = np.zeros(len(va_real), np.float32)

    print('[2/3] trening K-fold')
    for k in range(a.folds):
        pt = os.path.join(a.out, f'unet_fold{k}.pt')
        te_idx = np.nonzero(fold_of == k)[0]
        tr_idx = np.nonzero((fold_of >= 0) & (fold_of != k))[0]
        smp = X[tr_idx[:400]]
        mu = smp.mean((0, 2, 3), dtype=np.float64).reshape(1, -1, 1, 1).astype(np.float32)
        sd = (smp.std((0, 2, 3), dtype=np.float64) + 1e-4).reshape(1, -1, 1, 1).astype(np.float32)
        if os.path.exists(pt):
            net = E.PatchUNet(cin=X.shape[1], w=a.width).to(dev)
            net.load_state_dict(torch.load(pt, map_location=dev, weights_only=True))
            print(f'   fold {k}: wczytano {pt}')
        else:
            print(f'   fold {k}: trening {len(tr_idx)} / OOF {len(te_idx)}')
            net = train(a, dev, X, rows, tr_idx, mu, sd, pt, a.seed + 10 * k)
        # OOF na te_idx
        for s in range(0, len(te_idx), 500):
            b = te_idx[s:s + 500]
            p, g = infer([net], dev, X[b], mu, sd)
            P_all[b] = p.astype(np.float16); G_all[b] = g
        # wklad do va/test (srednia)
        pv, gv = infer([net], dev, X[va_real], mu, sd)
        P_va += pv / a.folds; G_va += gv / a.folds
        for s in range(0, len(Xt), 500):
            b = np.arange(s, min(s + 500, len(Xt)))
            p, g = infer([net], dev, Xt[b], mu, sd)
            P_te[b] += p / a.folds; G_te[b] += g / a.folds
        del net; torch.cuda.empty_cache()

    P_all[va_real] = P_va.astype(np.float16); G_all[va_real] = G_va
    P_all.flush(); G_all.flush()
    np.savez_compressed(os.path.join(a.out, 'meta.npz'),
                        va_real=va_real, fold_of=fold_of,
                        ids_train=np.array([r['id'] for r in rows]),
                        roots=np.array([r['root'] for r in rows]))
    np.savez_compressed(os.path.join(a.out, 'P_test.npz'), P=P_te.astype(np.float16), G=G_te)
    np.savez_compressed('cache_v2/unet_oof_val.npz', P=P_va.astype(np.float16), G=G_va)
    np.savez_compressed('cache_v2/unet_oof_test.npz', P=P_te.astype(np.float16), G=G_te)

    print('[3/3] walidacja OOF')
    report('walidacja prawdziwa OOF', P_va, [rows[i]['box'] for i in va_real], c_tr[va_real])
    print(f'zapisano {a.out}/P_train.f16.npy ({len(rows)} obrazkow) i cache_v2/unet_oof_*.npz')


if __name__ == '__main__':
    main()
