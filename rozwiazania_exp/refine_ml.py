"""
refine_ml.py - madrzejszy wybor krawedzi ramki (zamiast reguly z refine_v2).

Dla kazdej ramki z gotowego zgloszenia sprawdzamy ok. 2000 wariantow (kazda krawedz
-3..+3 px) i model (gradient boosting) przewiduje IoU kazdego wariantu z prawdziwa latka.
Wybieramy wariant z najwyzszym przewidywanym IoU.

Model uczymy na TWOICH wygenerowanych obrazkach z synth_v2, ktorych train_v3 nie widzial
(z 55 tys. uzyl 15 tys.) - wiec mamy tysiace przykladow z prawdziwa ramka zamiast 400.
Uczciwy wynik podajemy na 400 prawdziwych obrazkach walidacji (model ich nie widzi).

Uruchamiac z folderu kolega:
    python solve_v2.py --unet unet,moj,unet3 --out sub_baza.csv        # decyzja ramka / 0,0,0,0
    python refine_ml.py --sub sub_baza.csv --out sub_ml.csv

Wszystkie etapy maja cache w cache_ref/ - po przerwaniu uruchom ta sama komende.
"""
import argparse, csv, os, sys, time
import numpy as np
import torch
sys.path.insert(0, '.'); sys.path.insert(0, 'exp')
import ensemble4 as E
import solve_v2 as S
import refine_v2 as R
import train_v3 as T3
from common import punkty, NONE, D, train_rows, test_ids, integral, best_rect
from causal_cache import compute_causal

ap = argparse.ArgumentParser()
ap.add_argument('--sub', required=True, help='zgloszenie z solve_v2.py (decyzja ramka / 0,0,0,0)')
ap.add_argument('--out', default='sub_ml.csv')
ap.add_argument('--synth', default='../bumas1/synth_v2')
ap.add_argument('--synth_used', type=int, default=50000, help='ile obrazkow z synth wzial NAJWIEKSZY train_v3 (--synth_max)')
ap.add_argument('--n_syn', type=int, default=2000, help='ile nowych obrazkow z latka do nauki modelu')
ap.add_argument('--unet', default='unet,unet3a,unet3', help='mapy do cech (bez moj - ona widziala synth_v2)')
ap.add_argument('--v3', default='unet3a=cache_v3,unet3=cache_v3_full', help='nazwa map v3 = folder z jej sieciami')
ap.add_argument('--start_unet', default='unet,moj,unet3a,unet3', help='mapy, z ktorych powstala ramka startowa (jak w --sub)')
ap.add_argument('--v3_width', type=int, default=96)
ap.add_argument('--off', default='-3,-2,-1,0,1,2,3')
ap.add_argument('--cache', default='cache_ref')
ap.add_argument('--no_val_fit', action='store_true', help='do testu NIE dokladaj walidacji do nauki')
a = ap.parse_args()
R.OFF = tuple(int(v) for v in a.off.split(','))
dev = 'cuda'
os.makedirs(a.cache, exist_ok=True)
tags = a.unet.split(',')
V3 = dict(kv.split('=') for kv in a.v3.split(','))
TAGS = a.unet.replace(',', '_')
KEY = f'u{a.synth_used}_{TAGS}'
T0 = time.time()
log = lambda s: print(f'[{(time.time() - T0) / 60:5.1f} min] {s}', flush=True)


# ------------------------------------------------------------------ cechy wariantow
def box_sums(I, Rr):
    return I[Rr[:, 3], Rr[:, 2]] - I[Rr[:, 1], Rr[:, 2]] - I[Rr[:, 3], Rr[:, 0]] + I[Rr[:, 1], Rr[:, 0]]


def var_feats(rects, drops, p1, cz, b0):
    """Cechy kazdego wariantu ramki (wiersz = wariant)."""
    Rr = np.array(rects, np.int64)
    e = np.stack([np.maximum(Rr[:, 0] - 2, 0), np.maximum(Rr[:, 1] - 2, 0),
                  np.minimum(Rr[:, 2] + 2, 64), np.minimum(Rr[:, 3] + 2, 64)], 1)
    w = Rr[:, 2] - Rr[:, 0]; h = Rr[:, 3] - Rr[:, 1]; ar = (w * h).astype(np.float64)
    ear = ((e[:, 2] - e[:, 0]) * (e[:, 3] - e[:, 1])).astype(np.float64)
    pc = p1 * cz
    out = []
    for m in (pc, p1, cz):
        I = integral(m.astype(np.float64)); A = float(m.sum())
        s = box_sums(I, Rr); se = box_sums(I, e)
        ring = (se - s) / np.maximum(ear - ar, 1)
        out += [s / (ar + A - s + 1e-9), s / ar, ring, s / ar - ring]
    a0 = (b0[2] - b0[0]) * (b0[3] - b0[1])
    is0 = np.array([r == tuple(b0) for r in rects], np.float64)
    d0 = drops[is0 > 0][0] if is0.any() else drops.max()
    rank = np.argsort(np.argsort(drops)) / max(len(drops) - 1, 1)
    F = np.stack(out + [
        drops, drops - drops.max(), drops - d0, rank, np.tanh(drops),
        ar / 100, ar / a0, w / h,
        Rr[:, 0] - b0[0], Rr[:, 1] - b0[1], Rr[:, 2] - b0[2], Rr[:, 3] - b0[3], is0,
        np.full(len(Rr), d0), np.full(len(Rr), a0 / 100)], 1)
    return F.astype(np.float32)


def build_table(bb, U8, P1, C, B0, path, gts=None, log_name=''):
    """-> lista (rects, F, iou|None); cache w path (npz)."""
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return list(z['T'])
    T = []
    t = time.time()
    for n, (u, p, c, b0) in enumerate(zip(U8, P1, C, B0)):
        cz = np.clip(np.asarray(c, np.float32).mean(0), 0, None); cz = cz / (cz.max() + 1e-9)
        rects = R.variants(tuple(int(v) for v in b0))
        dr = R.rect_margins(bb, u, rects, dev)
        F = var_feats(rects, dr, np.asarray(p, np.float32), cz, tuple(int(v) for v in b0))
        y = None if gts is None else np.array([punkty(gts[n], r) for r in rects], np.float32)
        T.append((np.array(rects, np.int16), F, y))
        if (n + 1) % 100 == 0 or n + 1 == len(U8):
            log(f'warianty {log_name} {n + 1}/{len(U8)}  ({time.time() - t:.0f}s)')
    arr = np.empty(len(T), object); arr[:] = T
    np.savez(path, T=arr)
    return T


# ------------------------------------------------------------------ modele
bb = E.load_backbone(D, dev)
rows_real = train_rows()
perm = np.random.default_rng(0).permutation(len(rows_real))
va = np.sort(perm[:500]); trs = np.sort(perm[500:])[:400]


@torch.no_grad()
def kolega_ctx():
    trs_u8 = np.stack([E.load_u8(D, 'train', rows_real[i]['id']) for i in trs])
    f = []
    for s in range(0, len(trs_u8), 64):
        x = torch.from_numpy(trs_u8[s:s + 64].transpose(0, 3, 1, 2).astype(np.float32) / 255).to(dev)
        f.append(E.build_features(bb, x)[0].cpu().numpy())
    smp = np.concatenate(f).astype(np.float16).astype(np.float32)
    mu = smp.mean((0, 2, 3), dtype=np.float64).reshape(1, -1, 1, 1).astype(np.float32)
    sd = (smp.std((0, 2, 3), dtype=np.float64) + 1e-4).reshape(1, -1, 1, 1).astype(np.float32)
    nets = []
    for k in range(1, 7):
        n = E.PatchUNet(w=96).to(dev)
        n.load_state_dict(torch.load(f'modele_natrenowane/unet_{k}.pt', map_location=dev, weights_only=True))
        nets.append(n.eval())
    return nets, mu, sd


def v3_ctx(folder):
    f_tr = np.load(os.path.join(folder, 'f_train.npy'), mmap_mode='r')
    c_tr = np.load(os.path.join(folder, 'causal_train.npy'), mmap_mode='r')
    smp = T3.Stack(f_tr, c_tr)[trs]
    mu = smp.mean((0, 2, 3), dtype=np.float64).reshape(1, -1, 1, 1).astype(np.float32)
    sd = (smp.std((0, 2, 3), dtype=np.float64) + 1e-4).reshape(1, -1, 1, 1).astype(np.float32)
    nets = []
    k = 1
    while os.path.exists(os.path.join(folder, f'unet3_{k}.pt')):
        n = E.PatchUNet(cin=26, w=a.v3_width).to(dev)
        n.load_state_dict(torch.load(os.path.join(folder, f'unet3_{k}.pt'), map_location=dev, weights_only=True))
        nets.append(n.eval()); k += 1
    log(f'sieci v3 z {folder}: {len(nets)}')
    return nets, mu, sd


# ------------------------------------------------------------------ 1) nowe obrazki z synth_v2
log('[1/4] obrazki do nauki: synth_v2, ktorych train_v3 nie widzial')
ex = E.read_train(a.synth)
ps = np.random.default_rng(0).permutation(len(ex))      # ta sama permutacja co w train_v3
unused = [ex[i] for i in ps[a.synth_used:] if ex[i]['box'] != NONE][:a.n_syn]
log(f'   wybrano {len(unused)} obrazkow z latka (z {len(ps) - a.synth_used} niewykorzystanych)')
syn_npz = os.path.join(a.cache, f'syn_maps_{KEY}_{len(unused)}.npz')
if os.path.exists(syn_npz):
    z = np.load(syn_npz); P_syn, C_syn = z['P'], z['C']
    U_syn = np.stack([E.load_u8(r['root'], 'train', r['id']) for r in unused])
else:
    U_syn = np.stack([E.load_u8(r['root'], 'train', r['id']) for r in unused])
    C_syn, _ = compute_causal(bb, U_syn, os.path.join(a.cache, f'syn_causal_u{a.synth_used}_{len(unused)}.npy'), dev, 'synth')
    C_syn = np.asarray(C_syn, np.float32)
    Ps = []
    if 'unet' in tags:
        nets, mu, sd = kolega_ctx()
        with torch.no_grad():
            P, _ = S.unet_maps(dev, U_syn, nets, mu, sd, bb)
        Ps.append(P.astype(np.float32)); del nets
        log('   mapy 6 sieci kolegi gotowe')
    v3_tags = [t for t in tags if t in V3]
    if v3_tags:
        f_syn, _ = E.cache_split(bb, dev, U_syn, os.path.join(a.cache, f'syn_f_u{a.synth_used}_{len(unused)}.npy'), 128)
    for t in v3_tags:
        nets, mu, sd = v3_ctx(V3[t])
        with torch.no_grad():
            P, _ = T3.infer(nets, dev, T3.Stack(f_syn, C_syn), mu, sd)
        Ps.append(P.astype(np.float32)); del nets
        log(f'   mapy {t} gotowe')
    for t in tags:
        if t != 'unet' and t not in V3:
            raise SystemExit(f'nie umiem policzyc map {t} dla obrazkow synth (podaj w --v3 albo usun z --unet)')
    P_syn = np.mean(Ps, 0)
    np.savez(syn_npz, P=P_syn.astype(np.float16), C=C_syn.astype(np.float16))
P1_syn = np.asarray(P_syn[:, 1], np.float32)
B0_syn = []
for p, c in zip(P1_syn, C_syn):
    cz = np.clip(np.asarray(c, np.float32).mean(0), 0, None); cz /= cz.max() + 1e-9
    B0_syn.append(best_rect(p * cz)[0])
G_syn = [r['box'] for r in unused]
T_syn = build_table(bb, U_syn, P1_syn, C_syn, B0_syn, os.path.join(a.cache, f'syn_table_{KEY}_{len(unused)}.npz'),
                    G_syn, 'synth')
log(f'   synth: start IoU {np.mean([punkty(g, b) for g, b in zip(G_syn, B0_syn)]):.3f}'
    f'  wyrocznia {np.mean([y.max() for _, _, y in T_syn]):.3f}')

# ------------------------------------------------------------------ 2) walidacja (prawdziwa)
log('[2/4] walidacja: 400 prawdziwych obrazkow z latka')
gt = [rows_real[i]['box'] for i in va]
adv = np.array([g != NONE for g in gt])
Pv, _ = S.cached_unet('val', 'cpu', lambda: None, lambda: None, tags)
Pst, _ = S.cached_unet('val', 'cpu', lambda: None, lambda: None, a.start_unet.split(','))
Cv = np.load('cache_v2/causal_val.npy')[adv].astype(np.float32)
U_v = np.stack([E.load_u8(D, 'train', rows_real[i]['id']) for i in va[adv]])
B0_v = []
for p, c in zip(Pst[adv, 1].astype(np.float32), Cv):
    cz = np.clip(c.mean(0), 0, None); cz /= cz.max() + 1e-9
    B0_v.append(best_rect(p * cz)[0])
G_v = [g for g in gt if g != NONE]
T_v = build_table(bb, U_v, Pv[adv, 1].astype(np.float32), Cv, B0_v,
                  os.path.join(a.cache, f'val_table_{TAGS}__{a.start_unet.replace(",", "_")}.npz'), G_v, 'walidacja')

# ------------------------------------------------------------------ 3) model wyboru
log('[3/4] nauka modelu wyboru wariantu')
from sklearn.ensemble import HistGradientBoostingRegressor


def stack(T, rng, per=400):
    X, Y = [], []
    for rects, F, y in T:
        keep = np.unique(np.concatenate([rng.choice(len(F), min(per, len(F)), replace=False),
                                         np.argsort(-F[:, 12])[:50], np.nonzero(F[:, 24])[0]]))
        X.append(F[keep]); Y.append(y[keep])
    return np.concatenate(X), np.concatenate(Y)


def fit(T):
    X, Y = stack(T, np.random.default_rng(0))
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
                                      min_samples_leaf=50, l2_regularization=1.0, random_state=0)
    return m.fit(X, Y)


def pick(m, T):
    return [tuple(int(v) for v in rects[int(np.argmax(m.predict(F)))]) for rects, F, _ in T]


m_syn = fit(T_syn)
ch = pick(m_syn, T_v)
st = np.mean([punkty(g, b) for g, b in zip(G_v, B0_v)])
new = np.mean([punkty(g, b) for g, b in zip(G_v, ch)])
orc = np.mean([y.max() for _, _, y in T_v])
log(f'   WALIDACJA (uczciwie, model uczony tylko na synth): IoU start {st:.3f} -> wybor ML {new:.3f}'
    f'   (wyrocznia {orc:.3f})')
log(f'   przyblizony zysk wyniku: {0.8 * 0.92 * (new - st):+.3f}')

# ------------------------------------------------------------------ 4) test
log('[4/4] test')
ids = test_ids()
sub = {r['id']: tuple(int(r[k]) for k in ('x_min', 'y_min', 'x_max', 'y_max')) for r in csv.DictReader(open(a.sub))}
sel = [i for i, iid in enumerate(ids) if sub[iid] != NONE]
log(f'   ramek do dopracowania: {len(sel)}')
m_fin = m_syn if a.no_val_fit else fit(T_syn + T_v)
Pt, _ = S.cached_unet('test', 'cpu', lambda: None, lambda: None, tags)
Ct = np.load('cache_v2/causal_test.npy', mmap_mode='r')
import json
ck = os.path.join(a.cache, f'test_wybor_{os.path.basename(a.sub)}_{a.unet.replace(",", "_")}.json')
wyb = json.load(open(ck)) if os.path.exists(ck) else {}
t = time.time()
for n, i in enumerate(sel):
    iid = ids[i]
    if iid in wyb:
        continue
    b0 = sub[iid]
    c = np.asarray(Ct[i], np.float32)
    cz = np.clip(c.mean(0), 0, None); cz = cz / (cz.max() + 1e-9)
    rects = R.variants(b0)
    dr = R.rect_margins(bb, E.load_u8(D, 'test', iid), rects, dev)
    F = var_feats(rects, dr, Pt[i, 1].astype(np.float32), cz, b0)
    wyb[iid] = list(rects[int(np.argmax(m_fin.predict(F)))])
    if (n + 1) % 200 == 0 or n + 1 == len(sel):
        json.dump(wyb, open(ck, 'w'))
        log(f'   test {n + 1}/{len(sel)}  ({time.time() - t:.0f}s)')
ch_t = [tuple(int(v) for v in wyb[ids[i]]) for i in sel]
zm = 0
for i, b in zip(sel, ch_t):
    zm += b != sub[ids[i]]
    sub[ids[i]] = b
with open(a.out, 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['id', 'x_min', 'y_min', 'x_max', 'y_max'])
    for iid in ids:
        b = sub[iid]
        assert b == NONE or (0 <= b[0] < b[2] <= 64 and 0 <= b[1] < b[3] <= 64), b
        w.writerow([iid, *[int(v) for v in b]])
log(f'zapisano {a.out}: zmieniono {zm} z {len(sel)} ramek')
