"""Diagnostyka wytrenowanych U-Netow na odlozonych 500 obrazkach (ten sam podzial co solve/ensemble4)."""
import sys, time
import numpy as np
import torch
sys.path.insert(0, '.'); sys.path.insert(0, 'exp')
import ensemble4 as E
from common import punkty, NONE, jpeg_u8

dev = 'cuda'
D = '../bumas1/p1_student'
rows = E.read_train(D)
bb = E.load_backbone(D, dev)
perm = np.random.default_rng(0).permutation(len(rows))
va = np.sort(perm[:500]); tr = np.sort(perm[500:])
u8 = np.stack([E.load_u8(D, 'train', rows[i]['id']) for i in va])
q = np.random.default_rng(7).integers(75, 88, len(va))
u8j = np.stack([jpeg_u8(u, k) for u, k in zip(u8, q)])


def feats(imgs):
    out = []
    for s in range(0, len(imgs), 64):
        x = torch.from_numpy(imgs[s:s + 64].transpose(0, 3, 1, 2).astype(np.float32) / 255).to(dev)
        f, _ = E.build_features(bb, x); out.append(f.cpu().numpy().astype(np.float16))
    return np.concatenate(out)


trs = np.stack([E.load_u8(D, 'train', rows[i]['id']) for i in tr[:400]])
smp = feats(trs).astype(np.float32)
mu = smp.mean((0, 2, 3), dtype=np.float64).reshape(1, -1, 1, 1).astype(np.float32)
sd = (smp.std((0, 2, 3), dtype=np.float64) + 1e-4).reshape(1, -1, 1, 1).astype(np.float32)
nets = []
for k in range(1, 7):
    n = E.PatchUNet(w=96).to(dev); n.load_state_dict(torch.load(f'modele_natrenowane/unet_{k}.pt', weights_only=True)); nets.append(n)

gt = [rows[i]['box'] for i in va]
has = np.array([g != NONE for g in gt])
for name, imgs in (('oryginal', u8), ('JPEG q75-87', u8j)):
    f = feats(imgs)
    for which, ns in (('model1', nets[:1]), ('ensemble6', nets)):
        P = 0; G = 0
        for n in ns:
            p, g = E.infer_one(n, dev, f, np.arange(len(f)), mu, sd); P = P + p / len(ns); G = G + g / len(ns)
        top = [E.best_rect(p[1])[0] for p in P]
        cands = [E.candidates(p) for p in P]
        iou_top = np.array([punkty(g, b) for g, b in zip(gt, top)])[has]
        iou_or = np.array([max(punkty(g, c) for c in cs) for g, cs in zip(gt, cands)])[has]
        A = P[:, 1].sum((1, 2))
        if which == 'ensemble6':
            np.savez_compressed(f'exp/val_{name.split()[0]}.npz', P=P.astype(np.float16), G=G, u8=imgs, va=va)
        from sklearn.metrics import roc_auc_score
        print(f'{name:12s} {which:9s} IoU(top) {iou_top.mean():.3f}  IoU(wyrocznia kand.) {iou_or.mean():.3f}  '
              f'top~0 {(iou_top < 0.05).mean():.3f}  AUC glowa {roc_auc_score(has, G):.3f}  AUC masa {roc_auc_score(has, A):.3f}',
              flush=True)
