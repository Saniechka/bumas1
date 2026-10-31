import csv, io, os, re, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

D = 'data/p1_student'
NONE = (0, 0, 0, 0)


def punkty(prawda, odp):
    if prawda == NONE:
        return 1.0 if odp == NONE else 0.0
    if odp == NONE:
        return 0.0
    w = max(0, min(prawda[2], odp[2]) - max(prawda[0], odp[0]))
    h = max(0, min(prawda[3], odp[3]) - max(prawda[1], odp[1]))
    wsp = w * h
    pole = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    return wsp / (pole(prawda) + pole(odp) - wsp)


def backbone(dev='cuda'):
    sys.path.insert(0, os.path.join(D, 'model'))
    from resnet import resnet18_tin
    m = resnet18_tin()
    m.load_state_dict(torch.load(f'{D}/model/resnet18_tin.pt', weights_only=True))
    m.eval().to(dev)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def train_rows():
    wn = [l.strip() for l in open(f'{D}/model/classes.txt') if l.strip()]
    widx = {w: i for i, w in enumerate(wn)}
    out = []
    for r in csv.DictReader(open(f'{D}/train_labels.csv')):
        m = re.search(r'true=(n\d+)', r['comment'])
        out.append(dict(id=r['id'], comment=r['comment'],
                        box=tuple(int(r[k]) for k in ('x_min', 'y_min', 'x_max', 'y_max')),
                        decoys=[tuple(map(int, g.groups())) for g in
                                re.finditer(r'at \[(\d+),(\d+),(\d+),(\d+)\]', r['comment'])],
                        true=widx.get(m.group(1), -1) if m else -1))
    return out


def test_ids():
    return [r['id'] for r in csv.DictReader(open(f'{D}/test_ids.csv'))]


def load_imgs(ids, split):
    return np.stack([np.asarray(Image.open(f'{D}/{split}/images/{i}.png').convert('RGB')) for i in ids])


def jpeg_u8(u8, q):
    b = io.BytesIO(); Image.fromarray(u8).save(b, 'JPEG', quality=int(q)); b.seek(0)
    return np.asarray(Image.open(b).convert('RGB'))


def to_t(u8):
    return torch.from_numpy(np.ascontiguousarray(u8)).permute(0, 3, 1, 2).float() / 255


def margin(z, c):
    zc = z.gather(1, c[:, None])[:, 0]
    zo = z.clone(); zo.scatter_(1, c[:, None], -1e4)
    return zc - zo.max(1).values


SHAPES = [(w, h) for w in range(8, 46) for h in range(8, 46)
          if 0.07 <= w * h / 4096 <= 0.165 and 0.45 <= w / h <= 2.2]


def integral(a):
    return np.pad(np.cumsum(np.cumsum(a, 0), 1), ((1, 0), (1, 0)))


def best_rect(p, A=None):
    """Ramka maksymalizujaca oczekiwane IoU dla mapy prawdopodobienstwa p."""
    I = integral(p.astype(np.float64))
    A = float(p.sum()) if A is None else float(A)
    best, bb = -1.0, (0, 0, 20, 20)
    for (w, h) in SHAPES:
        s = I[h:, w:] - I[:-h, w:] - I[h:, :-w] + I[:-h, :-w]
        v = s / (w * h + A - s + 1e-9)
        j = int(np.argmax(v))
        if v.ravel()[j] > best:
            best = float(v.ravel()[j])
            yy, xx = np.unravel_index(j, s.shape)
            bb = (int(xx), int(yy), int(xx + w), int(yy + h))
    return bb, best
