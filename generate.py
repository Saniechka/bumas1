"""
Generator WLASNYCH danych treningowych: sami atakujemy ResNet tak, jak zrobil to
organizator, ale z rozkladem dopasowanym do TESTU:
  - prostokaty w dowolnym miejscu obrazka (w treningu organizatora tylko w dolnej polowie),
  - JPEG o jakosci 70-95 (w treningu 90-95, w tescie czesc ~75-87),
  - 4 ataki z maska: PGD L_inf (bez celu i z celem), MI-FGSM, C&W L2,
  - minimalne epsilon, przy ktorym atak dziala PO zapisie do JPEG,
  - wabiki: szum +-eps (jednostajny, gaussowski, rademacher), ktory NIE zmienia odpowiedzi.

Wynik ma ten sam format co dane organizatora, wiec solve.py czyta go flaga --extra.

    pip install torch numpy pillow
    python generate.py --data /sciezka/p1_student --out synth --per_base 6
    python solve.py   --data /sciezka/p1_student --extra synth

Bazy: 500 czystych obrazkow (CLEAN) i 500 z samymi wabikami (DECOY) z treningu.
Na GPU 1000 baz x 6 wariantow = 6000 obrazkow w ok. 20-40 minut.
"""
import argparse, csv, io, os, re, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

EPS_GRID = [2, 3, 4, 5, 6, 8, 10, 13, 16, 20]          # /255, rosnaco -> minimalne dzialajace


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


def jpeg_t(x, qs):
    """Prawdziwy JPEG na batchu (N,3,64,64) w [0,1]; bez gradientu."""
    u = (x.detach().clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    out = np.stack([jpeg_u8(a, q) for a, q in zip(u, qs)])
    return torch.from_numpy(out).to(x.device).permute(0, 3, 1, 2).float() / 255.


def through_jpeg(x, qs):
    """BPDA (Athalye i in. 2018): w przod prawdziwy JPEG, wstecz gradient tozsamosciowy.
    Organizator tez liczyl gradient przez JPEG, wiec latka przetrwa zapis."""
    return x + (jpeg_t(x, qs) - x).detach()


def rand_rect(rng, taken):
    for _ in range(200):
        area = rng.uniform(0.08, 0.15) * 4096
        ar = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))
        w = int(round(np.sqrt(area * ar))); h = int(round(area / w))
        if not (4 <= w <= 64 and 4 <= h <= 64):
            continue
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


def attack(model, x, y, tgt, M, eps, kind, qs, steps):
    """Atak ograniczony maska M. Zwraca obraz po ataku (bez JPEG)."""
    delta = torch.zeros_like(x)
    if kind == 'cw':
        # C&W L2: min ||d||^2 + c * max(Z_y - max_{j!=y} Z_j, -kappa),  |d| <= 20/255
        w = torch.zeros_like(x, requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.01)
        c = eps * 255 / 2.0                       # eps w siatce sluzy tu jako sila (stala c)
        for _ in range(steps):
            d = torch.tanh(w) * (20 / 255) * M
            z = model(through_jpeg((x + d).clamp(0, 1), qs))
            zy = z.gather(1, y[:, None])[:, 0]
            zo = z.clone(); zo.scatter_(1, y[:, None], -1e9)
            loss = (d ** 2).sum((1, 2, 3)) + c * (zy - zo.max(1).values).clamp_min(-0.5)
            opt.zero_grad(); loss.sum().backward(); opt.step()
        return (x + torch.tanh(w.detach()) * (20 / 255) * M).clamp(0, 1)
    alpha = 2.5 * eps / steps if kind != 'mifgsm' else eps / steps
    if kind.startswith('pgd'):
        delta = (torch.rand_like(x) * 2 - 1) * eps * M
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
            delta = (delta + alpha * g.sign()).clamp(-eps, eps) * M
            delta = ((x + delta).clamp(0, 1) - x)
    return (x + delta.detach()).clamp(0, 1)


def decoy_noise(rng, b, eps):
    w, h = b[2] - b[0], b[3] - b[1]
    kind = rng.choice(['uniform', 'gaussian', 'rademacher'])
    if kind == 'uniform':
        n = rng.uniform(-eps, eps, (h, w, 3))
    elif kind == 'gaussian':
        n = np.clip(rng.normal(0, eps / 2, (h, w, 3)), -eps, eps)
    else:
        n = rng.choice([-eps, eps], (h, w, 3))
    return kind, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='synth')
    ap.add_argument('--per_base', type=int, default=6)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    model = load_backbone(a.data, dev)
    os.makedirs(os.path.join(a.out, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(a.out, 'model'), exist_ok=True)
    wnids = [l.strip() for l in open(os.path.join(a.data, 'model', 'classes.txt')) if l.strip()]

    rows = list(csv.DictReader(open(os.path.join(a.data, 'train_labels.csv'))))
    bases = []
    for r in rows:
        if r['comment'].startswith(('CLEAN', 'DECOY')):
            dec = [tuple(int(g) for g in m.groups())
                   for m in re.finditer(r'at \[(\d+),(\d+),(\d+),(\d+)\]', r['comment'])]
            u8 = np.asarray(Image.open(os.path.join(a.data, 'train', 'images', r['id'] + '.png'))
                            .convert('RGB'), np.uint8)
            bases.append((r['id'], u8, dec))
    print(f'baz: {len(bases)}, urzadzenie: {dev}')

    # plan: kazda baza x per_base; 80% latka, 10% tylko wabiki, 10% czyste (jak w zadaniu)
    jobs = []
    for bid, u8, dec in bases:
        for k in range(a.per_base):
            jobs.append(dict(bid=bid, u8=u8, dec=list(dec), k=k,
                             kind=rng.choice(['patch'] * 8 + ['decoy', 'clean']),
                             q=int(rng.integers(70, 96)),
                             atk=rng.choice(['pgd_u', 'pgd_t', 'mifgsm', 'cw'])))
    out_rows, t0 = [], time.time()
    fh = open(os.path.join(a.out, 'train_labels.csv'), 'w', newline='')
    wr = csv.writer(fh); wr.writerow(['id', 'split', 'usage', 'tier', 'x_min', 'y_min', 'x_max', 'y_max', 'comment'])

    for s in range(0, len(jobs), a.bs):
        J = jobs[s:s + a.bs]
        # 1. prostokaty: latka + nowe wabiki, bez nakladania na istniejace wabiki
        for j in J:
            taken = list(j['dec'])
            j['patch'] = rand_rect(rng, taken) if j['kind'] == 'patch' else None
            if j['patch']: taken.append(j['patch'])
            j['new_dec'] = []
            for _ in range(int(rng.integers(0, 3)) if j['kind'] != 'clean' else 0):
                b = rand_rect(rng, taken)
                if b: taken.append(b); j['new_dec'].append(b)
            if j['kind'] == 'decoy' and not j['new_dec'] and not j['dec']:
                b = rand_rect(rng, taken)
                if b: j['new_dec'].append(b)
        x = torch.from_numpy(np.stack([j['u8'] for j in J])).to(dev).permute(0, 3, 1, 2).float() / 255.
        with torch.no_grad():
            y = model(x).argmax(1)                  # na czystej bazie przewidziana = prawdziwa
        tgt = (y + torch.randint(1, 200, y.shape, device=dev)) % 200
        qs = [j['q'] for j in J]
        res = [None] * len(J)

        # 2. latki: minimalne eps z siatki, przy ktorym atak dziala po zapisie JPEG
        todo = [i for i, j in enumerate(J) if j['patch']]
        for e in EPS_GRID:
            if not todo:
                break
            for atk in ('pgd_u', 'pgd_t', 'mifgsm', 'cw'):
                idx = [i for i in todo if J[i]['atk'] == atk]
                if not idx:
                    continue
                M = mask_of([J[i]['patch'] for i in idx], dev)
                steps = 20 if atk == 'mifgsm' else 100
                xa = attack(model, x[idx], y[idx], tgt[idx], M, e / 255, atk, [qs[i] for i in idx], steps)
                u = (xa * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
                with torch.no_grad():
                    sv = torch.from_numpy(np.stack([jpeg_u8(u[n], qs[i]) for n, i in enumerate(idx)]))
                    pz = model(sv.to(dev).permute(0, 3, 1, 2).float() / 255.).argmax(1).cpu().numpy()
                for n, i in enumerate(idx):
                    if pz[n] != int(y[i]):            # dziala po zapisie -> to jest minimalne eps
                        res[i] = (u[n], e)
            todo = [i for i in todo if res[i] is None]
        # 3. wabiki + JPEG + kontrola: latka myli model, wabiki nie
        def pred_of(u8):
            with torch.no_grad():
                return int(model(torch.from_numpy(u8).to(dev).permute(2, 0, 1)[None].float() / 255.).argmax())
        for i, j in enumerate(J):
            if j['patch'] and res[i] is None:
                continue
            e_dec = (res[i][1] if j['patch'] else int(rng.choice(EPS_GRID))) / 255.
            base = j['u8'].astype(np.float64) / 255.
            img = (res[i][0] if j['patch'] else j['u8']).astype(np.float64) / 255.
            only_dec = base.copy(); desc = []
            for bx in j['new_dec']:
                kind, n = decoy_noise(rng, bx, e_dec)
                sl = (slice(bx[1], bx[3]), slice(bx[0], bx[2]))
                img[sl] = np.clip(img[sl] + n, 0, 1)
                only_dec[sl] = np.clip(only_dec[sl] + n, 0, 1)
                desc.append(f'{kind} eps={round(e_dec * 255)}/255 at [{bx[0]},{bx[1]},{bx[2]},{bx[3]}]')
            to_u8 = lambda v: (v * 255).round().astype(np.uint8)
            saved = jpeg_u8(to_u8(img), j['q'])
            clean_pred = int(y[i])
            if j['new_dec'] and pred_of(jpeg_u8(to_u8(only_dec), j['q'])) != clean_pred:
                continue                              # wabik zmienil odpowiedz -> odrzuc
            if j['patch'] and pred_of(saved) == clean_pred:
                continue                              # latka nie dziala po zapisie -> odrzuc
            if not j['patch'] and pred_of(saved) != clean_pred:
                continue
            iid = f"syn{s + i:06d}"
            Image.fromarray(saved).save(os.path.join(a.out, 'train', 'images', iid + '.png'))
            b = j['patch'] or (0, 0, 0, 0)
            all_dec = [f'old at [{d[0]},{d[1]},{d[2]},{d[3]}]' for d in j['dec']] + desc
            head = (f"ADV {j['atk']} eps={res[i][1]}/255 true={wnids[clean_pred]}" if j['patch']
                    else ('DECOY;' if all_dec else 'CLEAN;'))
            com = f"{head}; decoys: {len(all_dec)} {', '.join(all_dec)}; jpeg={j['q']}"
            wr.writerow([iid, 'train', '', 'synthetic', *b, com]); out_rows.append(iid)
        fh.flush()
        print(f'{min(s + a.bs, len(jobs))}/{len(jobs)}  zapisano {len(out_rows)}  {time.time() - t0:.0f}s', flush=True)
    fh.close()
    import shutil
    shutil.copy(os.path.join(a.data, 'model', 'classes.txt'), os.path.join(a.out, 'model', 'classes.txt'))
    print('gotowe:', len(out_rows), 'obrazkow w', a.out)


if __name__ == '__main__':
    main()
