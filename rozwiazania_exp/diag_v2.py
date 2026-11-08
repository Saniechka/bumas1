"""
diag_v2.py - gdzie solve_v2.py traci punkty na walidacji (500 odlozonych obrazkow).
Korzysta tylko z gotowego cache (cache_v2), liczy sie kilkanascie sekund.

    python diag_v2.py --unet unet,moj,unet3
"""
import argparse, os, re, sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'exp')
import solve_v2 as S
from common import punkty, NONE, train_rows

ap = argparse.ArgumentParser()
ap.add_argument('--unet', default='unet,moj,unet3')
ap.add_argument('--alpha', type=float, default=1.0)
a = ap.parse_args()

rows = train_rows()
va = np.sort(np.random.default_rng(0).permutation(len(rows))[:500])
P, G = S.cached_unet('val', 'cpu', lambda: None, lambda: None, a.unet.split(','))
C = np.load('cache_v2/causal_val.npy').astype(np.float32)
St = np.load('cache_v2/causal_val_st.npy')
gt = [rows[i]['box'] for i in va]
bv, Fv = S.analyse(P, G, C, St, a.alpha)

# te same 5 czesci co w solve_v2.py -> odpowiedz dla kazdego obrazka
fold = np.random.default_rng(0).integers(0, 5, len(va))
out = [None] * len(va)
for k in range(5):
    tr, te = np.nonzero(fold != k)[0], np.nonzero(fold == k)[0]
    o, _, _ = S.fit_and_score(Fv[tr], [gt[i] for i in tr], [bv[i] for i in tr], Fv[te], [bv[i] for i in te])
    for j, i in enumerate(te):
        out[i] = o[j]

sc = np.array([punkty(g, o) for g, o in zip(gt, out)])
print(f'wynik walidacji: {sc.mean():.4f}   (--unet {a.unet})')

def info(i):
    c = rows[va[i]]['comment']
    atk = c.split()[1] if c.startswith('ADV') else '-'
    m = re.search(r'eps=(\d+)', c); eps = int(m.group(1)) if m and c.startswith('ADV') else 0
    m = re.search(r'jpeg=(\d+)', c); q = int(m.group(1)) if m else 0
    return atk, eps, q

has = [i for i in range(len(va)) if gt[i] != NONE]
no = [i for i in range(len(va)) if gt[i] == NONE]
dec = lambda i, b: any(punkty(d, b) >= 0.3 for d in rows[va[i]]['decoys'])

print(f'\n== bez latki ({len(no)} obrazkow, {100 * len(no) / len(va):.0f}%)')
fz = [i for i in no if out[i] != NONE]
print(f'   falszywa ramka: {len(fz)}  (w tym na wabiku: {sum(dec(i, out[i]) for i in fz)})'
      f'   -> strata {len(fz) / len(va):.3f} wyniku')

print(f'\n== z latka ({len(has)} obrazkow)')
grp = dict(zero=[], wabik=[], obok=[], trafil=[])
for i in has:
    o = out[i]
    if o == NONE: grp['zero'].append(i)
    elif punkty(gt[i], o) >= 0.1: grp['trafil'].append(i)
    elif dec(i, o): grp['wabik'].append(i)
    else: grp['obok'].append(i)
N = len(va)
for k, opis in (('trafil', 'trafiona latka'), ('zero', 'odpowiedz 0,0,0,0'),
                ('wabik', 'ramka na wabiku'), ('obok', 'ramka w zlym miejscu')):
    g = grp[k]
    if k == 'trafil':
        iou = np.mean([sc[i] for i in g]) if g else 0
        print(f'   {opis:22s}: {len(g):4d} ({100 * len(g) / len(has):4.1f}%)  srednie IoU {iou:.3f}'
              f'  -> strata przez niedokladna ramke {sum(1 - sc[i] for i in g) / N:.3f}')
    else:
        print(f'   {opis:22s}: {len(g):4d} ({100 * len(g) / len(has):4.1f}%)  -> strata {len(g) / N:.3f}')

def tab(nazwa, key):
    d = {}
    for i in has:
        d.setdefault(key(i), []).append(i)
    print(f'\n== wynik na obrazkach z latka wg {nazwa}')
    for k in sorted(d):
        g = d[k]
        print(f'   {str(k):10s} n={len(g):3d}  wynik {np.mean([sc[i] for i in g]):.3f}'
              f'   trafione {100 * np.mean([i in grp["trafil"] for i in g]):3.0f}%'
              f'   wabik {100 * np.mean([i in grp["wabik"] for i in g]):3.0f}%'
              f'   zero {100 * np.mean([i in grp["zero"] for i in g]):3.0f}%')

tab('ataku', lambda i: info(i)[0])
tab('eps (/255)', lambda i: (lambda e: '2-4' if e <= 4 else '5-8' if e <= 8 else '9-14' if e <= 14 else '15-20')(info(i)[1]))
tab('liczby wabikow', lambda i: len(rows[va[i]]['decoys']))
