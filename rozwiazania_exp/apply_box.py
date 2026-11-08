"""
apply_box.py - podmienia ramki w gotowym zgloszeniu na ramki z glosowania sieci v4
(albo na srednia obu). Decyzja ramka / 0,0,0,0 zostaje bez zmian.

Najpierw pokazuje na walidacji, ktory sposob jest najlepszy:
    python apply_box.py --sub sub_baza.csv --unet unet,moj,unet3 --out sub_box.csv --mode avg

--mode reg   ramka z glosowania v4
--mode avg   srednia wspolrzednych: ramka z map (jak w solve_v2) i z glosowania
--mode w     srednia wazona: --w 0..1 (waga glosowania)
"""
import argparse, csv, sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'exp')
import solve_v2 as S
from common import punkty, NONE, train_rows, test_ids, best_rect

ap = argparse.ArgumentParser()
ap.add_argument('--sub', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--unet', default='unet,moj,unet3', help='mapy, z ktorych powstaly ramki w --sub')
ap.add_argument('--tag', default='unet4')
ap.add_argument('--mode', default='avg', choices=['reg', 'avg', 'w'])
ap.add_argument('--w', type=float, default=0.5)
a = ap.parse_args()


def mix(b_map, b_reg, w):
    if b_reg[0] < 0:
        return b_map
    r = [int(round((1 - w) * u + w * v)) for u, v in zip(b_map, b_reg)]
    r[2] = max(r[2], r[0] + 4); r[3] = max(r[3], r[1] + 4)
    return tuple(min(max(v, 0), 64) for v in r)


# ---- walidacja: ktora waga najlepsza?
rows = train_rows()
va = np.sort(np.random.default_rng(0).permutation(len(rows))[:500])
gt = [rows[i]['box'] for i in va]
Pv, _ = S.cached_unet('val', 'cpu', lambda: None, lambda: None, a.unet.split(','))
Cv = np.load('cache_v2/causal_val.npy').astype(np.float32)
Bv = np.load(f'cache_v2/{a.tag}_box_val.npy')
res = {}
for w in (0.0, 0.25, 0.5, 0.75, 1.0):
    s = []
    for i, g in enumerate(gt):
        if g == NONE:
            continue
        cz = np.clip(Cv[i].mean(0), 0, None); cz /= cz.max() + 1e-9
        s.append(punkty(g, mix(best_rect(Pv[i, 1].astype(np.float32) * cz)[0], Bv[i], w)))
    res[w] = np.mean(s)
print('walidacja, IoU ramki na obrazkach z latka (waga glosowania v4):')
for w, v in res.items():
    print(f'   waga {w:.2f}: {v:.3f}' + ('   <- tylko mapy (jak teraz)' if w == 0 else '') +
          ('   <- tylko glosowanie' if w == 1 else ''))
best_w = max(res, key=res.get)
w = {'reg': 1.0, 'avg': 0.5, 'w': a.w}[a.mode]
print(f'najlepsza waga na walidacji: {best_w:.2f} ({res[best_w]:.3f});  uzywam: {w:.2f}')
print(f'przyblizony zysk wyniku wzgledem samych map: {0.8 * 0.92 * (res.get(w, res[best_w]) - res[0.0]):+.3f}')

# ---- test
ids = test_ids()
Bt = np.load(f'cache_v2/{a.tag}_box_test.npy')
sub = {r['id']: tuple(int(r[k]) for k in ('x_min', 'y_min', 'x_max', 'y_max')) for r in csv.DictReader(open(a.sub))}
zm = 0
with open(a.out, 'w', newline='') as fh:
    wr = csv.writer(fh); wr.writerow(['id', 'x_min', 'y_min', 'x_max', 'y_max'])
    for i, iid in enumerate(ids):
        b = sub[iid]
        if b != NONE:
            nb = mix(b, Bt[i], w); zm += nb != b; b = nb
        wr.writerow([iid, *b])
print(f'zapisano {a.out}: zmieniono {zm} ramek')
