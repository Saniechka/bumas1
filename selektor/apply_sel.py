import sys, numpy as np, pickle, glob, csv
from fitsel import *
items=load(['cf_v*.pkl'])
Xt=np.concatenate([X_of(d) for d in items]); yt=np.concatenate([d['y'] for d in items])
reg=HGR(max_iter=200,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1.0,random_state=0).fit(Xt,yt)
pt=[reg.predict(X_of(d)) for d in items]
cls=HGC(max_iter=150,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=20,random_state=0).fit(
    np.stack([imgX(d,p) for d,p in zip(items,pt)]),[d['gt']!=Z for d in items])
ids=test_ids(); T=[]
for f in sorted(glob.glob('cft_*.pkl')): T+=pickle.load(open(f,'rb'))
assert len({d['i'] for d in T})==5000, len(T)
T=list({d['i']:d for d in T}.values()); T.sort(key=lambda d:d['i'])
pr=[reg.predict(X_of(d)) for d in T]
q=np.array([cls.predict_proba(imgX(d,p)[None])[0,1] for d,p in zip(T,pr)])
J=[int(np.argmax(p)) for p in pr]; e=np.array([np.clip(0.6*p[j],0.05,1) for p,j in zip(pr,J)])
marg=q*e-(1-q)
def write(name,mask):
    with open(name,'w',newline='') as fh:
        w=csv.writer(fh); w.writerow(['id','x_min','y_min','x_max','y_max'])
        for d,j,m in zip(T,J,mask):
            b=d['C'][j] if m else Z
            assert b==Z or (0<=b[0]<b[2]<=64 and 0<=b[1]<b[3]<=64)
            w.writerow([ids[d['i']],*[int(v) for v in b]])
    print('%-28s ramek %.1f%%'%(name,100*np.mean(mask)))
write('sub_A_regula.csv',marg>0)
for r in (0.76,0.80,0.84):
    write('sub_B_ramki%d.csv'%int(r*100), marg>=np.quantile(marg,1-r))
