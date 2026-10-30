"""U-Net z solve.py (PyTorch) przepisany na NumPy + te same 24 cechy.
Gradienty modelu bierzemy z g_*.npy (ta sama definicja: d top1 i d (top1-top2))."""
import numpy as np
from scipy.ndimage import uniform_filter, median_filter, correlate
from ptload import load_state_dict
from resnet_np import _conv

KB = np.array([[-1., 2., -1.], [2., -4., 2.], [-1., 2., -1.]], np.float32) / 4.0


def _box(a, k):            # a (C,H,W); torch: pad reflect + avg_pool  == scipy 'mirror'
    return uniform_filter(a, size=(1, k, k), mode='mirror')


def _dct8():
    n = np.arange(8.)
    m = np.cos((2 * n + 1) * n[:, None] * np.pi / 16) * np.sqrt(2 / 8)
    m[0] /= np.sqrt(2)
    return m


_D = _dct8()


def _block_dct(g255):     # (64,64) -> (2,64,64)
    b = g255.reshape(8, 8, 8, 8).transpose(0, 2, 1, 3)
    ac = np.abs(_D @ b @ _D.T)
    hi = ac[..., 4:, 4:].mean((-2, -1)); nz = (ac[..., 3:, 3:] > 1.0).mean((-2, -1))
    return np.repeat(np.repeat(np.stack([hi, nz]), 8, 1), 8, 2)


def _corr(r, g, k):
    num = _box((r * g).sum(0, keepdims=True), k)
    a = _box((r * r).sum(0, keepdims=True), k); b = _box((g * g).sum(0, keepdims=True), k)
    return num / (np.sqrt(np.maximum(a, 0)) * np.sqrt(np.maximum(b, 0)) + 1e-12)


def features(img, g1, g2):
    """img (64,64,3) [0,1]; g1,g2 (3,64,64) -> (24,64,64) float32, jak build_features."""
    x = img.transpose(2, 0, 1).astype(np.float64)
    g1 = g1.astype(np.float64); g2 = g2.astype(np.float64)
    gray = x.mean(0, keepdims=True)
    r = x - median_filter(x, size=(1, 3, 3), mode='mirror')
    absr = _box(np.abs(r).mean(0, keepdims=True), 5)
    sg = np.sqrt(np.maximum(_box(gray * gray, 9) - _box(gray, 9) ** 2, 1e-8))
    kb = correlate(gray[0], KB.astype(np.float64), mode='mirror')[None]
    dct = _block_dct(gray[0] * 255.0)
    rc = r - r.mean(0, keepdims=True)
    xc = _box(np.abs(rc).mean(0, keepdims=True), 5)
    sat = _box(((x < 0.02) | (x > 0.98)).astype(np.float64).mean(0, keepdims=True), 5)
    f = np.concatenate([
        x, r * 20, absr * 20, absr / (sg + 0.01), _box(np.abs(kb), 3) * 20, dct, xc * 20,
        g1 / (np.abs(g1).max() + 1e-12), g2 / (np.abs(g2).max() + 1e-12),
        _corr(r, g1, 5), _corr(r, g2, 5), _corr(r, g2, 9),
        _box(np.abs(g2).sum(0, keepdims=True), 5) / (np.abs(g2).sum(0).max() + 1e-12),
        _box((r * np.sign(g2)).sum(0, keepdims=True), 5) * 20, sat], 0)
    return np.clip(f, -1e3, 1e3).astype(np.float16).astype(np.float32)


def _gn(x, w, b, G=8, eps=1e-5):
    N, C, H, W = x.shape
    y = x.reshape(N, G, -1)
    m = y.mean(-1, keepdims=True); v = y.var(-1, keepdims=True)
    y = ((y - m) / np.sqrt(v + eps)).reshape(N, C, H, W)
    return y * w.reshape(1, -1, 1, 1) + b.reshape(1, -1, 1, 1)


def _silu(x):
    return x / (1 + np.exp(-x))


def _pool(x):
    N, C, H, W = x.shape
    return x.reshape(N, C, H // 2, 2, W // 2, 2).mean((3, 5))


def _up(x):               # bilinear x2, align_corners=False
    def ax(a, axis):
        n = a.shape[axis]
        lo = np.take(a, np.clip(np.arange(n) - 1, 0, n - 1), axis)
        hi = np.take(a, np.clip(np.arange(n) + 1, 0, n - 1), axis)
        even = 0.25 * lo + 0.75 * a; odd = 0.75 * a + 0.25 * hi
        out = np.stack([even, odd], axis + 1)
        sh = list(a.shape); sh[axis] = 2 * n
        return out.reshape(sh)
    return ax(ax(x, 2), 3)


class UNetNP:
    def __init__(self, path):
        sd = load_state_dict(path)
        self.sd = {k: v.astype(np.float32) for k, v in sd.items()}

    def _blk(self, x, p):
        s = self.sd
        for c, g in ((0, 1), (3, 4)):
            w = s[f'{p}.{c}.weight']
            x = _conv(x, w, np.zeros(w.shape[0], np.float32), 1)
            x = _silu(_gn(x, s[f'{p}.{g}.weight'], s[f'{p}.{g}.bias']))
        return x

    def __call__(self, x):
        s = self.sd
        e1 = self._blk(x, 'e1'); e2 = self._blk(_pool(e1), 'e2')
        e3 = self._blk(_pool(e2), 'e3'); e4 = self._blk(_pool(e3), 'e4')
        h = e4.mean((2, 3))
        h = _silu(h @ s['head.0.weight'].T + s['head.0.bias']) @ s['head.2.weight'].T + s['head.2.bias']
        d3 = self._blk(np.concatenate([_up(e4), e3], 1), 'd3')
        d2 = self._blk(np.concatenate([_up(d3), e2], 1), 'd2')
        d1 = self._blk(np.concatenate([_up(d2), e1], 1), 'd1')
        o = _conv(d1, s['out.weight'], s['out.bias'], 1)
        return o, h[:, 0]


def d4(a, k):
    if k & 4: a = np.swapaxes(a, -1, -2)
    if k & 1: a = a[..., :, ::-1]
    if k & 2: a = a[..., ::-1, :]
    return np.ascontiguousarray(a)


def d4_inv(a, k):
    if k & 2: a = a[..., ::-1, :]
    if k & 1: a = a[..., :, ::-1]
    if k & 4: a = np.swapaxes(a, -1, -2)
    return np.ascontiguousarray(a)


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def infer(net, X, tta=8):
    """X (N,24,64,64) znormalizowane -> P (N,3,64,64), logit 'czy latka' (N,)."""
    acc = 0; g = 0
    for k in range(tta):
        o, h = net(d4(X, k))
        acc = acc + d4_inv(softmax(o), k); g = g + h
    return acc / tta, g / tta
