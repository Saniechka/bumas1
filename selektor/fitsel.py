import sys, numpy as np, pickle, glob
sys.path.insert(0,'/tmp/claude-0/-home-claude/608ece64-4ebb-5e4c-80b1-9cfa20ce448a/scratchpad/stub'); sys.path.insert(0,'solution_best')
import solve as S
from common import *
from sklearn.ensemble import HistGradientBoostingRegressor as HGR, HistGradientBoostingClassifier as HGC
rows=train_rows(); Z=(0,0,0,0)
def load(pats):
    it=[]
    for f in sorted(sum([glob.glob(p) for p in pats],[])):
        for d in pickle.load(open(f,'rb')):
            r=rows[d['i']]; gt=tuple(int(r[k]) for k in ('x_min','y_min','x_max','y_max'))
            d['gt']=gt; d['y']=np.array([S.punkty(gt,c) if gt!=Z else 0. for c in d['C']]); d['jpeg']='cf_j' in f
            it.append(d)
    return it
def X_of(d):
    F=d['F']; im=np.repeat(d['imf'][None],len(F),0)
    rk=np.argsort(np.argsort(-F[:,0]))[:,None].astype(np.float32)
    return np.concatenate([F,F-F.mean(0,keepdims=True),im,rk],1)
def imgX(d,pred):
    j=int(np.argmax(pred)); s=np.sort(pred)
    return np.concatenate([d['imf'],d['F'][j],[pred.max(), s[-2] if len(s)>1 else 0, pred.sum()]])
def evaluate(items,folds=5,seed=0,kap=None):
    grp=np.array([d['i'] for d in items]); ug=np.unique(grp)
    fold=dict(zip(ug,np.random.default_rng(seed).integers(0,folds,len(ug))))
    fi=np.array([fold[g] for g in grp])
    pred=[None]*len(items); ph=np.zeros(len(items))
    for k in range(folds):
        tr=[n for n in range(len(items)) if fi[n]!=k]; te=[n for n in range(len(items)) if fi[n]==k]
        Xt=np.concatenate([X_of(items[n]) for n in tr]); yt=np.concatenate([items[n]['y'] for n in tr])
        reg=HGR(max_iter=200,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1.0,random_state=0).fit(Xt,yt)
        # out-of-fold predykcje dla treningu klasyfikatora (wewnetrzne, prosto: in-fold)
        pt={n:reg.predict(X_of(items[n])) for n in tr}
        cls=HGC(max_iter=150,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=20,random_state=0).fit(
            np.stack([imgX(items[n],pt[n]) for n in tr]),[items[n]['gt']!=Z for n in tr])
        for n in te:
            pred[n]=reg.predict(X_of(items[n])); ph[n]=cls.predict_proba(imgX(items[n],pred[n])[None])[0,1]
    out={}
    for kp in ([kap] if kap else [0.6,0.8,1.0,1.2,1.5]):
        sc=[]
        for d,p,q in zip(items,pred,ph):
            j=int(np.argmax(p)); e=np.clip(kp*p[j],0.05,1)
            sc.append(S.punkty(d['gt'], d['C'][j] if q*e>1-q else Z))
        out[kp]=np.array(sc)
    sel=np.array([d['y'][int(np.argmax(p))] for d,p in zip(items,pred) if d['gt']!=Z])
    return out, sel.mean()
def baseline(items):
    its=[]
    for d in items:
        F=d['F']; C=np.stack([F[:,0],F[:,2],F[:,10],F[:,15],F[:,16]],1).astype(np.float64)
        im=dict(g=float(d['imf'][8]),A=float(d['imf'][9]),dall=float(d['imf'][2]),fa=float(d['imf'][3]))
        its.append((C,im,d['C'],d['gt']))
    return S.tune(its)['cv']
if __name__=='__main__':
    for name,pats in [('oryginal',['cf_v*.pkl']),('JPEG',['cf_j*.pkl']),('razem',['cf_v*.pkl','cf_j*.pkl'])]:
        it=load(pats)
        if not it: continue
        out,sel=evaluate(it)
        best=max(out,key=lambda k:out[k].mean())
        print('%-9s n=%d  TWOJA metoda (CV): %.4f   NOWA (CV): %s   IoU wybranej latki %.3f, wyrocznia %.3f'%(
            name,len(it),baseline(it),' '.join('k%.1f=%.4f'%(k,v.mean()) for k,v in out.items()),sel,
            np.mean([d['y'].max() for d in it if d['gt']!=Z])))
