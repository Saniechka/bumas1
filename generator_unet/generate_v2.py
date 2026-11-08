"""
generate_v2.py — poprawiony generator danych treningowych do zadania P1.

Co zmienione wzgledem generate.py (dlaczego — w README_v2.md):
  1. C&W szuka NAJMNIEJSZEJ dzialajacej normy L2 (wyszukiwanie binarne stalej c,
     zapamietanie najmniejszego udanego szumu). Stara wersja robila jeden przebieg
     ze stala c=1 i 95% latek C&W dostawalo opis eps=2 - szum nie byl minimalny.
     Teraz domyslnie: 100 krokow (jak w opisie), kappa=0, sprawdzanie co 5 krokow
     PO zapisie JPEG i zapamietanie najmniejszego udanego szumu.
  2. Bazy z walidacji (te same 500 obrazkow, ktore solve.py odklada) NIE sa uzywane.
     Inaczej wynik walidacji jest zawyzony (siec widziala te obrazki w treningu).
  3. Wiecej roznorodnych baz: domyslnie losowe odbicie lewo-prawo PRZED atakiem
     (--sym all = 8 symetrii, ale obrocone sceny sa nienaturalne dla ResNetu).
  4. Wabiki: jednostajny, gaussowski i rademacher (+-eps) - organizator uzywa wszystkich
     trzech. Sila losowana niezaleznie od sily latki.
  5. eps L_inf: najmniejsza CALKOWITA wartosc 1..20 (wyszukiwanie binarne zamiast
     rzadkiej siatki), id bazy w komentarzu.
  6. Sila i liczba wabikow losowane z PRAWDZIWEGO treningu organizatora.
  7. Polowa obrazkow z JPEG 90-95 (jak trening organizatora), polowa 70-89
     (test bywa trudniejszy). Prostokaty sprawdzane po zaokragleniu (8-15%, 1:2-2:1).

    python generate_v2.py --data D:/sciezka/p1_student --out synth_v2 --per_base 8
    python solve.py --data D:/sciezka/p1_student --extra synth_v2
"""
import argparse, csv, io, os, re, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

EPS_GRID = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20]          # /255, rosnaco
CW_MAX = 20 / 255


def load_backbone(data, dev):
    sys.path.insert(0, os.path.join(data, 'model'))
    from resnet import resnet18_tin
    m = resnet18_tin()
    m.load_state_dict(torch.load(os.path.join(data, 'model', 'resnet18_tin.pt'),
                                 map_location='cpu', weights_only=True))
    m.eval().to(dev)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def jpeg_u8(u8, q):
    buf = io.BytesIO(); Image.fromarray(u8).save(buf, 'JPEG', quality=int(q)); buf.seek(0)
    return np.asarray(Image.open(buf).convert('RGB'), np.uint8)


def to_u8(x):
    return (x.detach().clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def jpeg_t(x, qs):
    out = np.stack([jpeg_u8(a, q) for a, q in zip(to_u8(x), qs)])
    return torch.from_numpy(out).to(x.device).permute(0, 3, 1, 2).float() / 255.


def through_jpeg(x, qs):
    """BPDA: w przod prawdziwy JPEG, wstecz gradient tozsamosciowy."""
    return x + (jpeg_t(x, qs) - x).detach()


@torch.no_grad()
def pred_saved(model, x, qs):
    """Klasa modelu na obrazku po zapisie (zaokraglenie do uint8 + JPEG)."""
    return model(jpeg_t(x, qs)).argmax(1)


# ------------------------------------------------------------- symetrie bazy
def d4_img(u8, k):
    if k & 4: u8 = u8.transpose(1, 0, 2)
    if k & 1: u8 = u8[:, ::-1]
    if k & 2: u8 = u8[::-1, :]
    return np.ascontiguousarray(u8)


def d4_box(b, k):
    x0, y0, x1, y1 = b
    if k & 4: x0, y0, x1, y1 = y0, x0, y1, x1
    if k & 1: x0, x1 = 64 - x1, 64 - x0
    if k & 2: y0, y1 = 64 - y1, 64 - y0
    return (x0, y0, x1, y1)


# ------------------------------------------------------------- prostokaty
def rand_rect(rng, taken):
    for _ in range(200):
        area = rng.uniform(0.08, 0.15) * 4096
        ar = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))
        w = int(round(np.sqrt(area * ar))); h = int(round(area / w))
        if not (4 <= w <= 64 and 4 <= h <= 64):
            continue
        if not (0.08 <= w * h / 4096 <= 0.15 and 0.5 <= w / h <= 2.0):
            continue                                    # po zaokragleniu poza zakresem z opisu
        x0 = int(rng.integers(0, 65 - w)); y0 = int(rng.integers(0, 65 - h))
        b = (x0, y0, x0 + w, y0 + h)
        if all(b[2] <= t[0] or t[2] <= b[0] or b[3] <= t[1] or t[3] <= b[1] for t in taken):
            return b
    return None


def mask_of(boxes, dev):
    m = torch.zeros(len(boxes), 1, 64, 64, device=dev)
    for i, b in enumerate(boxes):
        m[i, :, b[1]:b[3], b[0]:b[2]] = 1
    return m


# ------------------------------------------------------------- ataki L_inf
def attack_linf(model, x, y, tgt, M, eps, kind, qs, steps):
    """eps: tensor (N,1,1,1) - kazdy obrazek moze miec inne eps."""
    alpha = 2.5 * eps / steps if kind != 'mifgsm' else eps / steps
    delta = (torch.rand_like(x) * 2 - 1) * eps * M if kind.startswith('pgd') else torch.zeros_like(x)
    g_mom = torch.zeros_like(x)
    for _ in range(steps):
        delta.requires_grad_(True)
        z = model(through_jpeg((x + delta).clamp(0, 1), qs))
        loss = -F.cross_entropy(z, tgt) if kind == 'pgd_t' else F.cross_entropy(z, y)
        g = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            if kind == 'mifgsm':
                g_mom = g_mom + g / (g.abs().mean((1, 2, 3), keepdim=True) + 1e-12)
                g = g_mom
            delta = torch.maximum(torch.minimum(delta + alpha * g.sign(), eps), -eps) * M
            delta = (x + delta).clamp(0, 1) - x
    return (x + delta.detach()).clamp(0, 1)


# ------------------------------------------------------------- C&W L2 (minimalne)
def attack_cw(model, x, y, M, qs, steps=100, bsearch=1, c0=1.0, check_every=5, kappa=0.0):
    """Najmniejsza dzialajaca norma L2, |d| <= 20/255 tylko w masce.
    Domyslnie 1 przebieg x 100 krokow (jak w opisie zadania i w torchattacks)
    z zapamietaniem najmniejszego udanego szumu. bsearch>1 = dodatkowe
    wyszukiwanie binarne stalej c (oryginal Carlini-Wagner, latki trudniejsze)."""
    n = len(x)
    c = torch.full((n,), c0, device=x.device)
    lo = torch.zeros(n, device=x.device); hi = torch.full((n,), 1e9, device=x.device)
    best_l2 = torch.full((n,), 1e9, device=x.device); best = x.clone()
    for _ in range(bsearch):
        w = torch.zeros_like(x, requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.01)
        ok_round = torch.zeros(n, dtype=torch.bool, device=x.device)
        for it in range(steps):
            d = torch.tanh(w) * CW_MAX * M
            z = model(through_jpeg((x + d).clamp(0, 1), qs))
            zy = z.gather(1, y[:, None])[:, 0]
            zo = z.clone(); zo.scatter_(1, y[:, None], -1e9)
            f = (zy - zo.max(1).values).clamp_min(-kappa)
            l2 = (d ** 2).sum((1, 2, 3))
            loss = (l2 + c * f).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            if (it + 1) % check_every == 0:
                with torch.no_grad():
                    xa = (x + torch.tanh(w) * CW_MAX * M).clamp(0, 1)
                    ok = pred_saved(model, xa, qs) != y           # sukces PO zapisie
                    l2n = ((xa - x) ** 2).sum((1, 2, 3))
                    better = ok & (l2n < best_l2)
                    best[better] = xa[better]; best_l2[better] = l2n[better]
                    ok_round |= ok
        with torch.no_grad():                                   # binarne c
            hi = torch.where(ok_round, torch.minimum(hi, c), hi)
            lo = torch.where(ok_round, lo, torch.maximum(lo, c))
            c = torch.where(hi < 1e8, (lo + hi) / 2, c * 10)
    found = best_l2 < 1e8
    return best, found


def decoy_noise(rng, b, eps, allow_rad, g_sd=0.5):
    w, h = b[2] - b[0], b[3] - b[1]
    kinds = ['uniform', 'gaussian'] + (['rademacher'] if allow_rad else [])
    kind = rng.choice(kinds)
    if kind == 'uniform':
        n = rng.uniform(-eps, eps, (h, w, 3))
    elif kind == 'gaussian':
        n = np.clip(rng.normal(0, eps * g_sd, (h, w, 3)), -eps, eps)
    else:
        n = rng.choice([-eps, eps], (h, w, 3))
    return kind, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='synth_v2')
    ap.add_argument('--per_base', type=int, default=8)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--holdout', type=int, default=500, help='tyle samo co w solve.py')
    ap.add_argument('--q_min', type=int, default=70)
    ap.add_argument('--q_max', type=int, default=95)
    ap.add_argument('--no_rademacher', action='store_true',
                    help='bez wabikow +-eps (organizator ich UZYWA - ok. 35%% wabikow w treningu)')
    ap.add_argument('--p_patch', type=float, default=0.86,
                    help='czesc zadan z latka; czesc atakow odpada, wiec po odrzuceniu wychodzi ok. 80%%')
    ap.add_argument('--sym', default='flip', choices=['none', 'flip', 'all'],
                    help='flip = tylko odbicie lewo-prawo (naturalne obrazki); all = 8 symetrii')
    ap.add_argument('--q_hi_frac', type=float, default=0.5,
                    help='czesc obrazkow z JPEG 90-95 jak w treningu organizatora; reszta q_min..89')
    ap.add_argument('--cw_bsearch', type=int, default=1, help='1 = jak w opisie (100 krokow)')
    ap.add_argument('--gauss_sd', type=float, default=0.5, help='odchylenie wabika gaussowskiego / eps')
    ap.add_argument('--limit', type=int, default=0, help='tylko pierwsze N zadan (test)')
    ap.add_argument('--part', type=int, default=-1,
                    help='numer czesci (0,1,2,...): inny seed i osobny CSV; potem polacz.py')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if a.part >= 0:
        a.seed += 1000 * a.part
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    model = load_backbone(a.data, dev)
    os.makedirs(os.path.join(a.out, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(a.out, 'model'), exist_ok=True)
    wnids = [l.strip() for l in open(os.path.join(a.data, 'model', 'classes.txt')) if l.strip()]

    rows = list(csv.DictReader(open(os.path.join(a.data, 'train_labels.csv'))))
    # te same obrazki, ktore solve.py odklada do walidacji -> nie wolno ich uzyc
    val_ids = {rows[i]['id'] for i in np.random.default_rng(0).permutation(len(rows))[:a.holdout]}
    bases, skipped = [], 0
    for r in rows:
        if r['comment'].startswith(('CLEAN', 'DECOY')):
            if r['id'] in val_ids:
                skipped += 1; continue
            dec = [tuple(int(g) for g in m.groups())
                   for m in re.finditer(r'at \[(\d+),(\d+),(\d+),(\d+)\]', r['comment'])]
            u8 = np.asarray(Image.open(os.path.join(a.data, 'train', 'images', r['id'] + '.png'))
                            .convert('RGB'), np.uint8)
            bases.append((r['id'], u8, dec))
    print(f'baz: {len(bases)} (pominieto {skipped} z walidacji), urzadzenie: {dev}')
    # sila wabika "z tego samego rozkladu co latki": bierzemy eps z prawdziwych latek
    real_eps = [int(m.group(1)) for r in rows if r['comment'].startswith('ADV')
                for m in [re.search(r'eps=(\d+)', r['comment'])] if m and 1 <= int(m.group(1)) <= 20]
    if not real_eps:
        real_eps = EPS_GRID
    print(f'eps wabikow losowane z {len(real_eps)} prawdziwych latek, mediana {int(np.median(real_eps))}/255')
    # liczba wabikow obok latki - jak w prawdziwym treningu
    n_dec_adv = [len(re.findall(r'at \[', r['comment'])) for r in rows if r['comment'].startswith('ADV')] or [0, 1, 2]

    jobs = []
    for bid, u8, dec in bases:
        for k in range(a.per_base):
            s = {'none': 0, 'flip': int(rng.integers(2)), 'all': int(rng.integers(8))}[a.sym]
            jobs.append(dict(bid=bid, sym=s, u8=d4_img(u8, s).copy(), dec=[d4_box(d, s) for d in dec],
                             kind=('patch' if rng.random() < a.p_patch
                                   else rng.choice(['decoy', 'clean'])),
                             q=int(rng.integers(90, 96)) if rng.random() < a.q_hi_frac
                               else int(rng.integers(a.q_min, 90)),
                             atk=rng.choice(['pgd_u', 'pgd_t', 'mifgsm', 'cw'])))
    order = rng.permutation(len(jobs)); jobs = [jobs[i] for i in order]
    if a.limit:
        jobs = jobs[:a.limit]
    stat = dict(ok=0, fail_atk=0, fail_dec=0)
    csv_name = f'train_labels_part{a.part}.csv' if a.part >= 0 else 'train_labels.csv'
    prefix = f'syn2_p{a.part}_' if a.part >= 0 else 'syn2_'
    fh = open(os.path.join(a.out, csv_name), 'w', newline='')
    wr = csv.writer(fh); wr.writerow(['id', 'split', 'usage', 'tier', 'x_min', 'y_min', 'x_max', 'y_max', 'comment'])
    t0 = time.time()

    for s in range(0, len(jobs), a.bs):
        J = jobs[s:s + a.bs]
        for j in J:
            taken = list(j['dec'])
            j['patch'] = rand_rect(rng, taken) if j['kind'] == 'patch' else None
            if j['patch']: taken.append(j['patch'])
            j['new_dec'] = []
            n_new = 0 if j['kind'] == 'clean' else max(0, int(rng.choice(n_dec_adv)) - len(j['dec']))
            for _ in range(n_new):
                b = rand_rect(rng, taken)
                if b: taken.append(b); j['new_dec'].append(b)
            if j['kind'] == 'decoy' and not j['new_dec'] and not j['dec']:
                b = rand_rect(rng, taken)
                if b: j['new_dec'].append(b)
        x = torch.from_numpy(np.stack([j['u8'] for j in J])).to(dev).permute(0, 3, 1, 2).float() / 255.
        with torch.no_grad():
            y = model(x).argmax(1)
        # klasa docelowa: losowo jedna z 2.-5. najbardziej prawdopodobnych (nie z 200).
        # Z losowej klasy z 200 PGD z celem potrzebowal duzo wiekszego eps niz u
        # organizatora (mediana 12-13 zamiast 8) i czesto sie nie udawal.
        with torch.no_grad():
            top = model(x).topk(5, dim=1).indices
        tgt = top[torch.arange(len(J), device=dev), torch.randint(1, 5, (len(J),), device=dev)]
        qs = [j['q'] for j in J]
        res = [None] * len(J)                       # (obraz uint8 PRZED JPEG, eps_inf w /255)

        # C&W: osobno, bez siatki eps
        idx = [i for i, j in enumerate(J) if j['patch'] and j['atk'] == 'cw']
        if idx:
            M = mask_of([J[i]['patch'] for i in idx], dev)
            xa, found = attack_cw(model, x[idx], y[idx], M, [qs[i] for i in idx], bsearch=a.cw_bsearch)
            u = to_u8(xa)
            for n, i in enumerate(idx):
                if found[n]:
                    linf = int(np.abs(u[n].astype(int) - J[i]['u8'].astype(int)).max())
                    res[i] = (u[n], max(1, linf))
        # L_inf: najmniejsze CALKOWITE eps (1..20)/255 - wyszukiwanie binarne,
        # osobno dla kazdego obrazka. Najpierw eps=20: jak nie dziala, odpada.
        cand = [i for i, j in enumerate(J) if j['patch'] and j['atk'] != 'cw']
        lo = {i: 1 for i in cand}; hi = {i: 20 for i in cand}
        def run(ids, e_of):
            for atk in ('pgd_u', 'pgd_t', 'mifgsm'):
                idx = [i for i in ids if J[i]['atk'] == atk]
                if not idx:
                    continue
                M = mask_of([J[i]['patch'] for i in idx], dev)
                ev = torch.tensor([e_of[i] / 255 for i in idx], device=dev).view(-1, 1, 1, 1)
                q_idx = [qs[i] for i in idx]
                xa = attack_linf(model, x[idx], y[idx], tgt[idx], M, ev, atk, q_idx,
                                 20 if atk == 'mifgsm' else 100)
                pz = pred_saved(model, xa, q_idx).cpu().numpy(); u = to_u8(xa)
                for n, i in enumerate(idx):
                    ok = pz[n] != int(y[i])
                    if ok:
                        res[i] = (u[n], e_of[i]); hi[i] = e_of[i]
                    else:
                        lo[i] = e_of[i]
        run(cand, {i: 20 for i in cand})
        active = [i for i in cand if res[i] is not None]
        while True:
            active = [i for i in active if hi[i] - lo[i] > 1]
            if not active:
                break
            run(active, {i: (lo[i] + hi[i]) // 2 for i in active})

        def pred_of(u8):
            with torch.no_grad():
                return int(model(torch.from_numpy(np.ascontiguousarray(u8).copy()).to(dev).permute(2, 0, 1)[None].float() / 255.).argmax())
        for i, j in enumerate(J):
            if j['patch'] and res[i] is None:
                stat['fail_atk'] += 1; continue
            # sila wabika: niezalezna, z tego samego rozkladu co latki
            e_dec = int(rng.choice(real_eps)) / 255.
            base = j['u8'].astype(np.float64) / 255.
            img = (res[i][0] if j['patch'] else j['u8']).astype(np.float64) / 255.
            only_dec = base.copy(); desc = []
            for bx in j['new_dec']:
                kind, nz = decoy_noise(rng, bx, e_dec, not a.no_rademacher, a.gauss_sd)
                sl = (slice(bx[1], bx[3]), slice(bx[0], bx[2]))
                img[sl] = np.clip(img[sl] + nz, 0, 1)
                only_dec[sl] = np.clip(only_dec[sl] + nz, 0, 1)
                desc.append(f'{kind} eps={round(e_dec * 255)}/255 at [{bx[0]},{bx[1]},{bx[2]},{bx[3]}]')
            u8f = lambda v: (v * 255).round().astype(np.uint8)
            saved = jpeg_u8(u8f(img), j['q'])
            clean_pred = int(y[i])
            if j['new_dec'] and pred_of(jpeg_u8(u8f(only_dec), j['q'])) != clean_pred:
                stat['fail_dec'] += 1; continue
            if j['patch'] and pred_of(saved) == clean_pred:
                stat['fail_atk'] += 1; continue
            if not j['patch'] and pred_of(saved) != clean_pred:
                stat['fail_dec'] += 1; continue
            iid = f"{prefix}{s + i:06d}"
            Image.fromarray(saved).save(os.path.join(a.out, 'train', 'images', iid + '.png'))
            b = j['patch'] or (0, 0, 0, 0)
            all_dec = [f'old at [{d[0]},{d[1]},{d[2]},{d[3]}]' for d in j['dec']] + desc
            head = (f"ADV {j['atk']} eps={res[i][1]}/255 true={wnids[clean_pred]}" if j['patch']
                    else ('DECOY;' if all_dec else 'CLEAN;'))
            com = f"{head}; decoys: {len(all_dec)} {', '.join(all_dec)}; jpeg={j['q']}; base={j['bid']} sym={j['sym']}"
            wr.writerow([iid, 'train', '', 'synthetic', *b, com]); stat['ok'] += 1
        fh.flush()
        done = min(s + a.bs, len(jobs)); el = time.time() - t0
        eta = el / done * (len(jobs) - done)
        print(f'{done}/{len(jobs)}  {stat}  {el / 60:.1f} min, zostalo ok. {eta / 3600:.1f} h', flush=True)
    fh.close()
    import shutil
    shutil.copy(os.path.join(a.data, 'model', 'classes.txt'), os.path.join(a.out, 'model', 'classes.txt'))
    print('gotowe:', stat, '->', a.out)


if __name__ == '__main__':
    main()
