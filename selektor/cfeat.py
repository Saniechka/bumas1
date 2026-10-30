"""Cechy kandydatow: mapa U-Neta + test przyczynowy (3 filtry) + zgodnosc szumu z gradientem."""
import numpy as np, io
from PIL import Image
from scipy.ndimage import median_filter, gaussian_filter, uniform_filter
from gradcorr import noise_residual, corr_map

def rejpeg(u8,q):
    b=io.BytesIO(); Image.fromarray(u8).save(b,'JPEG',quality=int(q)); b.seek(0)
    return np.asarray(Image.open(b).convert('RGB'),np.uint8)

def _soft(z): e=np.exp(z-z.max(-1,keepdims=True)); return e/e.sum(-1,keepdims=True)

def _integral(a): return np.pad(np.cumsum(np.cumsum(a,0),1),((1,0),(1,0)))
def _bs(I,b): return I[b[3],b[2]]-I[b[1],b[2]]-I[b[3],b[0]]+I[b[1],b[0]]

def cand_features(net, img, g1, g2, P, ghead, cands):
    """img (64,64,3) [0,1]. Zwraca (len(cands), F) i cechy obrazka."""
    filt=[median_filter(img,size=(3,3,1)), np.round(img*15)/15, gaussian_filter(img,(1,1,0))]
    var=[img]+filt
    for b in cands:
        for f in filt:
            v=img.copy(); v[b[1]:b[3],b[0]:b[2]]=f[b[1]:b[3],b[0]:b[2]]; var.append(v)
    lg=net(np.stack([v.transpose(2,0,1) for v in var]).astype(np.float32)); p=_soft(lg)
    c0=int(lg[0].argmax()); l0=lg[0,c0]; srt=np.sort(lg[0])
    glob=[l0, srt[-1]-srt[-2]]
    for k in range(3):
        glob+=[l0-lg[1+k,c0], float(lg[1+k].argmax()!=c0)]
    r=noise_residual(img).astype(np.float32)
    cm=[corr_map(r,g2.astype(np.float32),5), corr_map(r,g1.astype(np.float32),5)]
    ar=uniform_filter(np.abs(r).sum(0),5)
    maps=[P[1],P[2],cm[0],cm[1],ar]
    I=[_integral(m.astype(np.float64)) for m in maps]
    A1=float(P[1].sum()); F=[]
    for j,b in enumerate(cands):
        a=(b[2]-b[0])*(b[3]-b[1])
        X0,Y0,X1,Y1=max(b[0]-4,0),max(b[1]-4,0),min(b[2]+4,64),min(b[3]+4,64)
        ring=(X1-X0)*(Y1-Y0)-a
        f=[]
        for Ii in I:
            s=_bs(Ii,b); so=_bs(Ii,(X0,Y0,X1,Y1))-s
            f+=[s/a, s/a-so/max(ring,1)]
        s1=_bs(I[0],b); f+=[s1/(a+A1-s1+1e-9), a/4096., (b[2]-b[0])/(b[3]-b[1]), float(j)]
        base=4+3*j
        for k in range(3):
            q=lg[base+k]
            dn=abs(l0-lg[1+k,c0])+0.5
            f+=[l0-q[c0], (l0-q[c0])/dn, float(q.argmax()!=c0), float(q.argmax()==lg[1+k].argmax()), np.abs(p[base+k]-p[0]).sum()]
        F.append(f)
    imf=glob+[float(ghead), A1/600., float(P[2].sum())/600., float(len(cands))]
    return np.array(F,np.float32), np.array(imf,np.float32)
