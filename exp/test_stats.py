"""Czym test rozni sie od treningu? Szacowana jakosc JPEG i kruchosc odpowiedzi modelu."""
import sys, re, collections
import numpy as np
import torch
sys.path.insert(0, 'exp')
from common import *

dev = 'cuda'
bb = backbone(dev)
rows = train_rows()
tr = load_imgs([r['id'] for r in rows], 'train')
te = load_imgs(test_ids(), 'test')
Q = list(range(60, 101, 1))


def est_q(u8):
    x = u8.astype(np.float32)
    err = [np.abs(jpeg_u8(u8, q).astype(np.float32) - x).mean() for q in Q]
    return Q[int(np.argmin(err))], min(err)


@torch.no_grad()
def flips(U, q):
    out = []
    for s in range(0, len(U), 256):
        x = to_t(U[s:s + 256]).to(dev); f = to_t(np.stack([jpeg_u8(u, q) for u in U[s:s + 256]])).to(dev)
        with torch.autocast('cuda', dtype=torch.float16):
            out.append((bb(x).argmax(1) != bb(f).argmax(1)).cpu().numpy())
    return np.concatenate(out)


rng = np.random.default_rng(0)
itr = rng.choice(len(tr), 800, replace=False); ite = rng.choice(len(te), 800, replace=False)
qt = np.array([est_q(u)[0] for u in tr[itr]]); qe = np.array([est_q(u)[0] for u in te[ite]])
true_q = np.array([int(re.search(r'jpeg=(\d+)', rows[i]['comment']).group(1)) for i in itr])
print('trening: szacowana==prawdziwa', np.mean(np.abs(qt - true_q) <= 1))
h = lambda q: sorted(collections.Counter((q // 5 * 5).tolist()).items())
print('trening szac. q:', h(qt)); print('test     szac. q:', h(qe))
ft, fe = flips(tr, 50), flips(te, 50)
adv = np.array([r['box'] != NONE for r in rows])
print(f'JPEG50 calosci zmienia klase: trening {ft.mean():.3f} (adv {ft[adv].mean():.3f}, brak {ft[~adv].mean():.3f}), test {fe.mean():.3f}')
lo = ite[qe < 89]; hi = ite[qe >= 89]
print(f'   test q<89: n={len(lo)} zmiana {fe[lo].mean():.3f};  test q>=89: n={len(hi)} zmiana {fe[hi].mean():.3f}')
np.save('exp/test_q_sample.npy', np.stack([ite, qe]))
