"""U-Nety (ensemble 6) na probce testu: polozenie ramek i pewnosc w zaleznosci od jakosci JPEG."""
import sys
import numpy as np
import torch
sys.path.insert(0, '.'); sys.path.insert(0, 'exp')
import ensemble4 as E
from common import test_ids, load_imgs

dev = 'cuda'
D = '../bumas1/p1_student'
bb = E.load_backbone(D, dev)
ite, qe = np.load('exp/test_q_sample.npy')
ids = test_ids()
U = load_imgs([ids[i] for i in ite], 'test')
rows = E.read_train(D)
perm = np.random.default_rng(0).permutation(len(rows))


def feats(imgs):
    out = []
    for s in range(0, len(imgs), 64):
        x = torch.from_numpy(imgs[s:s + 64].transpose(0, 3, 1, 2).astype(np.float32) / 255).to(dev)
        f, _ = E.build_features(bb, x); out.append(f.cpu().numpy().astype(np.float16))
    return np.concatenate(out)


smp = feats(load_imgs([rows[i]['id'] for i in np.sort(perm[500:])[:400]], 'train')).astype(np.float32)
mu = smp.mean((0, 2, 3), dtype=np.float64).reshape(1, -1, 1, 1).astype(np.float32)
sd = (smp.std((0, 2, 3), dtype=np.float64) + 1e-4).reshape(1, -1, 1, 1).astype(np.float32)
f = feats(U)
P = 0; G = 0
for k in range(1, 7):
    n = E.PatchUNet(w=96).to(dev); n.load_state_dict(torch.load(f'modele_natrenowane/unet_{k}.pt', weights_only=True))
    p, g = E.infer_one(n, dev, f, np.arange(len(f)), mu, sd); P = P + p / 6; G = G + g / 6
np.savez_compressed('exp/test_sample_unet.npz', P=P.astype(np.float16), G=G, ite=ite, qe=qe)
pg = 1 / (1 + np.exp(-G))
box = np.array([E.best_rect(p[1])[0] for p in P])
cy = (box[:, 1] + box[:, 3]) / 2
for nm, m in (('q>=89', qe >= 89), ('q<89', qe < 89)):
    conf = pg[m] > 0.5
    print(f'{nm:6s} n={m.sum()}  P(latka)>0.5: {conf.mean():.3f}  sr. P(latka) {pg[m].mean():.3f}  '
          f'srodek ramki w gornej polowie: {(cy[m][conf] < 32).mean():.3f}  max p1 {P[m, 1].max((1, 2)).mean():.3f}')
