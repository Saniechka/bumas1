"""
porownaj_dane.py — czy wygenerowane obrazki sa podobne do prawdziwych z treningu?

Porownuje prawdziwy trening organizatora i Twoj folder z generatora:
  - udzialy CLEAN / DECOY / ADV i atakow,
  - rozklad eps dla kazdego ataku,
  - jakosc JPEG,
  - "widocznosc" latki: ile wiecej szumu wysokiej czestotliwosci jest w latce
    niz poza nia (|obraz - mediana 3x3|). To najwazniejsza liczba: jesli w Twoich
    danych latki sa duzo "glosniejsze" niz u organizatora, siec uczy sie zbyt
    latwego zadania i slabo dziala na tescie.

    python porownaj_dane.py --data D:/sciezka/p1_student --synth synth_v2
"""
import argparse, collections, csv, os, re
import numpy as np
from PIL import Image


def median3(x):
    p = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode='reflect')
    st = np.stack([p[i:i + 64, j:j + 64] for i in range(3) for j in range(3)])
    return np.median(st, 0)


def contrast(img, box):
    """srednia |residuum| w ramce / srednia |residuum| poza ramkami."""
    x = img.astype(np.float32)
    r = np.abs(x - median3(x)).mean(2)
    m = np.zeros((64, 64), bool); m[box[1]:box[3], box[0]:box[2]] = True
    return r[m].mean() / (r[~m].mean() + 1e-6)


def summary(root, n_img):
    rows = list(csv.DictReader(open(os.path.join(root, 'train_labels.csv'))))
    kinds = collections.Counter(r['comment'].split()[0].rstrip(';') for r in rows)
    dk = collections.Counter(); nd = collections.Counter()
    atk = collections.Counter(); eps = collections.defaultdict(list); q = []; cp = collections.defaultdict(list); cd = []
    rng = np.random.default_rng(0)
    pick = set(rng.choice(len(rows), min(n_img, len(rows)), replace=False))
    for k, r in enumerate(rows):
        c = r['comment']
        dk.update(re.findall(r'(uniform|gaussian|rademacher)', c))
        nd[len(re.findall(r'at \[', c))] += 1
        m = re.search(r'jpeg=(\d+)', c)
        if m: q.append(int(m.group(1)))
        if c.startswith('ADV'):
            a = c.split()[1]; atk[a] += 1
            e = re.search(r'eps=(\d+)', c)
            if e: eps[a].append(int(e.group(1)))
        if k in pick:
            img = np.asarray(Image.open(os.path.join(root, 'train', 'images', r['id'] + '.png')).convert('RGB'))
            b = tuple(int(r[t]) for t in ('x_min', 'y_min', 'x_max', 'y_max'))
            if b != (0, 0, 0, 0):
                cp[c.split()[1]].append(contrast(img, b))
            for d in re.finditer(r'at \[(\d+),(\d+),(\d+),(\d+)\]', c):
                cd.append(contrast(img, tuple(int(g) for g in d.groups())))
    return dict(n=len(rows), dk=dk, nd=nd, kinds=kinds, atk=atk, eps=eps, q=q, cp=cp, cd=cd)


def pr(name, s):
    print(f'\n===== {name}: {s["n"]} obrazkow')
    tot = sum(s['kinds'].values())
    print('  rodzaje :', ', '.join(f'{k} {100 * v / tot:.0f}%' for k, v in s['kinds'].most_common()))
    print('  typy wabikow:', dict(s['dk']), '  wabikow na obrazek:', dict(sorted(s['nd'].items())))
    ta = sum(s['atk'].values()) or 1
    print('  ataki   :', ', '.join(f'{k} {100 * v / ta:.0f}%' for k, v in sorted(s['atk'].items())))
    if s['q']:
        print(f'  JPEG q  : min {min(s["q"])}  mediana {int(np.median(s["q"]))}  max {max(s["q"])}')
    for a in sorted(s['eps']):
        e = np.array(s['eps'][a])
        print(f'  eps {a:7s}: mediana {np.median(e):4.1f}  25% {np.percentile(e, 25):4.1f}  75% {np.percentile(e, 75):4.1f}'
              f'  (eps=2: {100 * np.mean(e == 2):.0f}%)')
    print('  widocznosc latki (szum w ramce / poza ramka), mediana:')
    for a in sorted(s['cp']):
        print(f'     {a:7s}: {np.median(s["cp"][a]):.2f}   (n={len(s["cp"][a])})')
    if s['cd']:
        print(f'     wabiki : {np.median(s["cd"]):.2f}   (n={len(s["cd"])})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='folder p1_student (prawdziwe dane)')
    ap.add_argument('--synth', required=True, help='folder z generatora')
    ap.add_argument('--n', type=int, default=1500, help='ile obrazkow czytac do pomiaru widocznosci')
    a = ap.parse_args()
    pr('PRAWDZIWE (organizator)', summary(a.data, a.n))
    pr('WYGENEROWANE', summary(a.synth, a.n))
    print('\nJak czytac: liczby "widocznosc" powinny byc podobne w obu zbiorach dla kazdego ataku.'
          '\nDuzo wieksze w wygenerowanych = Twoje latki sa za latwe.')


if __name__ == '__main__':
    main()
