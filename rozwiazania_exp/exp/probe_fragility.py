"""Czy odszumienie samej łatki przywraca klasę, a wabika nie? (na prawdziwym treningu)"""
import csv, io, os, re, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

D = '../bumas1/p1_student'
dev = 'cuda'
sys.path.insert(0, os.path.join(D, 'model'))
from resnet import resnet18_tin
bb = resnet18_tin(); bb.load_state_dict(torch.load(f'{D}/model/resnet18_tin.pt', weights_only=True)); bb.eval().to(dev)
wn = [l.strip() for l in open(f'{D}/model/classes.txt') if l.strip()]; widx = {w: i for i, w in enumerate(wn)}
rows = list(csv.DictReader(open(f'{D}/train_labels.csv')))
X = np.stack([np.asarray(Image.open(f'{D}/train/images/{r["id"]}.png').convert('RGB')) for r in rows])
x = torch.from_numpy(X).permute(0, 3, 1, 2).float().div(255)


def med3(t):
    p = F.pad(t, (1, 1, 1, 1), mode='reflect')
    return p.unfold(2, 3, 1).unfold(3, 3, 1).reshape(*t.shape, 9).median(-1).values


def blur(t, s=1.0):
    k = torch.arange(-2, 3, dtype=t.dtype, device=t.device); g = torch.exp(-k ** 2 / (2 * s * s)); g /= g.sum()
    t = F.conv2d(F.pad(t, (2, 2, 0, 0), mode='reflect'), g.view(1, 1, 1, 5).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(t, (0, 0, 2, 2), mode='reflect'), g.view(1, 1, 5, 1).repeat(3, 1, 1, 1), groups=3)


def jpg(t, q):
    u = (t.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy(); o = []
    for a in u:
        b = io.BytesIO(); Image.fromarray(a).save(b, 'JPEG', quality=q); b.seek(0); o.append(np.asarray(Image.open(b)))
    return torch.from_numpy(np.stack(o)).to(t.device).permute(0, 3, 1, 2).float() / 255


@torch.no_grad()
def pred(t):
    out = []
    for i in range(0, len(t), 256):
        with torch.autocast('cuda', dtype=torch.float16):
            out.append(bb(t[i:i + 256].to(dev)).argmax(1).cpu())
    return torch.cat(out)


def chunked(f, t):
    return torch.cat([f(t[i:i + 256].to(dev)).cpu() for i in range(0, len(t), 256)])


t0 = time.time(); p0 = pred(x); print('forward 5000:', time.time() - t0)
true = np.array([widx.get(m.group(1), -1) if (m := re.search(r'true=(n\d+)', r['comment'])) else -1 for r in rows])
box = np.array([[int(r[k]) for k in ('x_min', 'y_min', 'x_max', 'y_max')] for r in rows])
dec = [[tuple(map(int, m.groups())) for m in re.finditer(r'at \[(\d+),(\d+),(\d+),(\d+)\]', r['comment'])] for r in rows]
adv = box[:, 2] > 0
p0c = p0.cpu().numpy()
print('latka myli model:', (p0c[adv] != true[adv]).mean())


def region(t, b):
    m = torch.zeros(1, 1, 64, 64); m[..., b[1]:b[3], b[0]:b[2]] = 1; return m


fil = dict(med3=chunked(med3, x), blur1=chunked(blur, x), blur07=chunked(lambda t: blur(t, 0.7), x),
           jpeg50=jpg(x, 50), jpeg75=jpg(x, 75), q16=(x * 15).round() / 15)
ia = np.nonzero(adv)[0]
idd = [(i, d) for i in range(len(rows)) for d in dec[i]]
for name, f in fil.items():
    M = torch.cat([region(x, box[i]) for i in ia])
    xa = x[ia] * (1 - M) + f[ia] * M
    pa = pred(xa).cpu().numpy()
    Md = torch.cat([region(x, d) for i, d in idd]); ii = np.array([i for i, _ in idd])
    pd = pred(x[ii] * (1 - Md) + f[ii] * Md).cpu().numpy()
    pall = pred(f).cpu().numpy()
    print(f'{name:7s} latka->zmiana {np.mean(pa != p0c[ia]):.3f}  latka->true {np.mean(pa == true[ia]):.3f}  '
          f'wabik->zmiana {np.mean(pd != p0c[ii]):.3f}  caly obraz->zmiana(adv) {np.mean(pall[ia] != p0c[ia]):.3f} '
          f'caly->zmiana(bez latki) {np.mean(pall[~adv] != p0c[~adv]):.3f}')
