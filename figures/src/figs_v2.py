import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({"font.family":"serif","font.serif":["Source Serif Pro","DejaVu Serif"],"font.size":9,
  "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#444","axes.linewidth":.6,"svg.fonttype":"none"})
INK="#1F3A5F"; G="#888"
def save(f,n): f.tight_layout(); f.savefig(f"figures/{n}.svg"); plt.close(f)
# 1 prior -> posterior
x=np.linspace(0,1,400); f,ax=plt.subplots(figsize=(6.4,2.4))
ax.plot(x,stats.beta(2,2).pdf(x),color=G,ls="--",lw=1,label="prior Beta(2, 2)")
ax.plot(x,stats.binom(120,x).pmf(37)*121,color=G,lw=1,label="likelihood (scaled), 37/120")
ax.plot(x,stats.beta(39,85).pdf(x),color=INK,lw=1.6,label="posterior Beta(39, 85)")
ax.set_xlabel("conversion rate p"); ax.set_yticks([]); ax.legend(frameon=False,fontsize=8); save(f,"v2_bayes_update")
# 2 CUPED
rng=np.random.default_rng(1); pre=rng.gamma(2,10,600); y=.8*pre+rng.normal(0,8,600)
f,ax=plt.subplots(1,2,figsize=(6.4,2.3)); ax[0].scatter(pre,y,s=3,color=G); ax[0].set_xlabel("pre-period metric X"); ax[0].set_ylabel("Y")
yc=y-np.cov(y,pre)[0,1]/pre.var()*(pre-pre.mean())
ax[1].hist(y-y.mean(),40,color="#CCC",label="Y"); ax[1].hist(yc-yc.mean(),40,color=INK,alpha=.8,label="Y CUPED"); ax[1].legend(frameon=False,fontsize=8); ax[1].set_yticks([])
save(f,"v2_cuped")
# 3 DAG-style confounding / backdoor
f,ax=plt.subplots(figsize=(6.4,1.9)); ax.axis("off")
P={"X (confounder)":(0.5,.85),"T (treatment)":(0.18,.15),"Y (outcome)":(0.82,.15),"M (mediator)":(0.5,.15)}
for k,(a,b) in P.items(): ax.text(a,b,k,ha="center",va="center",fontsize=9,bbox=dict(boxstyle="square,pad=.35",fc="white",ec="#444",lw=.6))
for s,t in [("X (confounder)","T (treatment)"),("X (confounder)","Y (outcome)"),("T (treatment)","M (mediator)"),("M (mediator)","Y (outcome)")]:
    ax.annotate("",xy=P[t],xytext=P[s],arrowprops=dict(arrowstyle="-|>",color=INK,lw=.8,shrinkA=22,shrinkB=22))
ax.set_xlim(0,1); ax.set_ylim(0,1); save(f,"v2_dag")
# 4 rolling-origin backtest
f,ax=plt.subplots(figsize=(6.4,1.8))
for i,o in enumerate(range(180,228,12)):
    ax.barh(i,o,color="#DDD",height=.55); ax.barh(i,12,left=o,color=INK,height=.55)
ax.set_yticks(range(4),[f"fold {i+1}" for i in range(4)]); ax.set_xlabel("time index (train grey, forecast horizon ink)"); save(f,"v2_backtest")
# 5 PSI
ref=rng.normal(0,1,20000); f,ax=plt.subplots(figsize=(6.4,2.1))
for c,lab,col in [(rng.normal(0,1,5000),"reference",G),(rng.normal(1,1,5000),"current (shift 1 sd)",INK)]:
    ax.hist(c,60,density=True,histtype="step",color=col,lw=1.2,label=lab)
ax.legend(frameon=False,fontsize=8); ax.set_yticks([]); save(f,"v2_drift")
# 6 skew
f,ax=plt.subplots(figsize=(6.4,2.0)); keys=rng.zipf(1.3,200000)%5000
import hashlib; from collections import Counter
h=lambda k:int(hashlib.md5(str(k).encode()).hexdigest(),16)%16
L=Counter(h(k) for k in keys); ax.bar(range(16),[L[i] for i in range(16)],color=INK,width=.7)
ax.axhline(len(keys)/16,color=G,ls="--",lw=.8); ax.set_xlabel("partition"); ax.set_ylabel("rows"); save(f,"v2_skew")
