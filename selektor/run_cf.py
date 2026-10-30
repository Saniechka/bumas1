"""split train|trainjpeg|test; liczy (jesli trzeba) gradienty, U-Net, kandydatow i cechy."""
import sys, numpy as np, pickle, time
sys.path.insert(0,'/tmp/claude-0/-home-claude/608ece64-4ebb-5e4c-80b1-9cfa20ce448a/scratchpad/stub'); sys.path.insert(0,'solution_best')
import solve as S
from common import *
from unet_np import UNetNP, features, infer
from resnet_np import ResNet18NP
from resnet_grad import ResNet18Grad
from cfeat import cand_features, rejpeg
split, lo, hi, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
idx=np.load('va_idx.npy') if split!='test' else np.arange(5000)
sel=idx[lo:hi]
mu,sd=np.load('unet_musd.npy'); unet=UNetNP('/mnt/user-data/uploads/bumas2/unet.pt')
net=ResNet18NP(ROOT+'model/resnet18_tin.pt')
if split=='trainjpeg': gnet=ResNet18Grad(ROOT+'model/resnet18_tin.pt')
G=np.load('g_%s.npy'%('test' if split=='test' else 'train'),mmap_mode='r')
ids=test_ids() if split=='test' else [r['id'] for r in train_rows()]
qs=np.random.default_rng(55).integers(70,89,5000)
res=[];t=time.time()
for i in sel:
    img=load_img(ids[i],'test' if split=='test' else 'train')
    if split=='trainjpeg':
        img=rejpeg((img*255).round().astype(np.uint8),qs[i]).astype(np.float32)/255.
        x=img.transpose(2,0,1)[None].astype(np.float32)
        lg,_,_=gnet.forward_cache(x); o=np.argsort(-lg[0])
        d1=np.zeros_like(lg); d1[0,o[0]]=1; _,g1=gnet.input_grad(x,d1)
        d2=np.zeros_like(lg); d2[0,o[0]]=1; d2[0,o[1]]=-1; _,g2=gnet.input_grad(x,d2)
        g1,g2=g1[0],g2[0]
    else:
        g1,g2=np.asarray(G[i,0],np.float32),np.asarray(G[i,1],np.float32)
    X=(features(img,g1,g2)[None]-mu)/sd
    P,gh=infer(unet,X); P=P[0]
    C=S.candidates(P)
    F,imf=cand_features(net,img,g1,g2,P,gh[0],C)
    res.append(dict(i=int(i),C=C,F=F,imf=imf,q=int(qs[i]) if split=='trainjpeg' else None))
    if len(res)==100:
        pickle.dump(res,open(out.replace('.pkl','_%04d.pkl'%(i)),'wb')); res=[]
if res: pickle.dump(res,open(out.replace('.pkl','_%04d.pkl'%(sel[-1])),'wb')); print('done',out,'%.2f s/img'%((time.time()-t)/len(sel)))
