import sys, itertools
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as HGR

for name in sys.argv[1:]:
    T = np.load(name)
    T = T[T[:, 9] >= 0]
    k = T[:, 0].astype(int)
    imgs = np.unique(k)
    groups = {i: np.nonzero(k == i)[0] for i in imgs}
    st = {i: g[T[g, 8] == 1][0] for i, g in groups.items()}

    def pick(score):
        return np.mean([T[g[np.argmax(score[g])], 9] for g in groups.values()])

    print(f'== {name}  obrazkow z latka: {len(imgs)}')
    print(f'  start (U-Net) {np.mean([T[st[i], 9] for i in imgs]):.3f}   wyrocznia {np.mean([T[g, 9].max() for g in groups.values()]):.3f}')
    a, md, mp, eu = T[:, 1], T[:, 2], T[:, 3], T[:, 6]
    best = None
    for al, be, ga in itertools.product((0, .2, .5, 1, 2, 4), (0, .25, .5, 1, 2), (0, 1, 2, 4, 8)):
        s = pick(-md - al * a / 100 + be * mp + ga * eu)
        if best is None or s > best[0]:
            best = (s, al, be, ga)
    print(f'  regula liniowa (strojona na tych samych danych) {best[0]:.3f}  alfa={best[1]} beta={best[2]} gamma={best[3]}')
    # GBM z cechami wzglednymi, CV po obrazkach
    s0 = np.array([st[i] for i in k])
    Xf = np.stack([a / 100, a / T[s0, 1], md, mp, md - T[s0, 2], mp - T[s0, 3], T[:, 4], T[:, 5], eu, T[:, 7],
                   T[:, 10] - T[s0, 10], T[:, 11] - T[s0, 11], T[:, 12] - T[s0, 12], T[:, 13] - T[s0, 13]], 1)
    fold = np.random.default_rng(0).integers(0, 5, imgs.max() + 1)[k]
    pred = np.zeros(len(T))
    for f in range(5):
        tr, te = fold != f, fold == f
        m = HGR(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50).fit(Xf[tr], T[tr, 9])
        pred[te] = m.predict(Xf[te])
    print(f'  GBM (5-fold CV) {pick(pred):.3f}')
