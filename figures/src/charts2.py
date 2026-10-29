from figstyle import *
import warnings; warnings.filterwarnings("ignore")
from sklearn.datasets import make_moons, make_classification, load_diabetes, load_breast_cancer
from sklearn.model_selection import learning_curve, train_test_split, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, lasso_path, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
rng=np.random.default_rng(0)
cm2=LinearSegmentedColormap.from_list("tA",["#FDE68A","#FFFFFF","#99F6E4"])

# decision boundaries
X,y=make_moons(400,noise=0.28,random_state=1);Xtr,Xte,ytr,yte=train_test_split(X,y,random_state=0)
models=[("Logistic regression",make_pipeline(StandardScaler(),LogisticRegression())),("k-NN (k=15)",make_pipeline(StandardScaler(),KNeighborsClassifier(15))),
("Decision tree (depth 6)",DecisionTreeClassifier(max_depth=6,random_state=0)),("Random forest",RandomForestClassifier(300,min_samples_leaf=3,random_state=0)),
("Gradient boosting",HistGradientBoostingClassifier(max_iter=200)),("SVM (RBF)",make_pipeline(StandardScaler(),SVC(C=3,probability=True))),
("Gaussian Naive Bayes",GaussianNB()),("MLP (2×32)",make_pipeline(StandardScaler(),MLPClassifier((32,32),max_iter=2000,random_state=0)))]
xx,yy=np.meshgrid(np.linspace(-1.8,2.8,220),np.linspace(-1.4,1.9,220))
f,axs=plt.subplots(2,4,figsize=(12,5.6))
for ax,(n,m) in zip(axs.flat,models):
    m.fit(Xtr,ytr);Z=m.predict_proba(np.c_[xx.ravel(),yy.ravel()])[:,1].reshape(xx.shape)
    ax.contourf(xx,yy,Z,levels=20,cmap=cm2,alpha=.9);ax.contour(xx,yy,Z,levels=[.5],colors=N,linewidths=1.4)
    ax.scatter(Xte[:,0],Xte[:,1],c=np.where(yte==1,T,A),s=9,edgecolor="white",lw=.3)
    ax.set_title(n,fontsize=9.5);ax.set_xticks([]);ax.set_yticks([]);ax.grid(False)
    ax.text(.97,.04,f"test acc {m.score(Xte,yte):.2f}",transform=ax.transAxes,ha="right",fontsize=8,weight="bold",color=N,bbox=dict(fc="white",ec="#CBD5E1",pad=1.5))
f.suptitle("Decision boundaries on the same noisy 'moons' data (black line = P(y=1)=0.5)",weight="bold",color=N);save(f,"boundaries")

# learning curves real
Xc,yc=make_classification(3000,20,n_informative=6,random_state=0)
f,axs=plt.subplots(1,2,figsize=(10,3.6))
for ax,(n,m) in zip(axs,[("Logistic regression → high bias",LogisticRegression(max_iter=500)),("Deep tree → high variance",DecisionTreeClassifier(random_state=0))]):
    sz,tr,va=learning_curve(m,Xc,yc,cv=5,train_sizes=np.linspace(.05,1,10),n_jobs=-1)
    for s,c,l in [(tr,T,"train"),(va,A,"cross-val")]:
        ax.plot(sz,s.mean(1),"o-",color=c,label=l,ms=4);ax.fill_between(sz,s.mean(1)-s.std(1),s.mean(1)+s.std(1),color=c,alpha=.15)
    ax.set_title(n);ax.set_xlabel("training examples");ax.set_ylabel("accuracy");ax.legend(loc="lower right");ax.set_ylim(.6,1.02)
axs[0].annotate("curves converge low:\nmore data won't help",xy=(2200,.84),xytext=(900,.68),fontsize=8,color=R,arrowprops=dict(arrowstyle="->",color=R))
axs[1].annotate("large gap: more data /\nregularisation will help",xy=(2000,.88),xytext=(700,.68),fontsize=8,color=R,arrowprops=dict(arrowstyle="->",color=R))
save(f,"learning_curves_real")

# validation curve (complexity)
d=np.arange(1,21);tr,va=validation_curve(DecisionTreeClassifier(random_state=0),Xc,yc,param_name="max_depth",param_range=d,cv=5,n_jobs=-1)
f,ax=plt.subplots(figsize=(6.5,3.3));ax.plot(d,1-tr.mean(1),"o-",ms=3,label="train error",color=T);ax.plot(d,1-va.mean(1),"o-",ms=3,label="validation error",color=A)
b=d[np.argmax(va.mean(1))];ax.axvline(b,ls="--",color=R);ax.text(b+.3,.25,f"best depth = {b}",color=R,weight="bold")
ax.axvspan(0.5,3.5,color="#FEF3C7",alpha=.6);ax.axvspan(12,20.5,color="#FCE7F3",alpha=.6);ax.text(1,.02,"underfit",color=A);ax.text(15,.02,"overfit",color=R)
ax.set_xlabel("max_depth (model complexity)");ax.set_ylabel("error");ax.legend();ax.set_title("Validation curve: error vs complexity (real CV results)");save(f,"validation_curve")

# regularization paths on diabetes
Xd,yd=load_diabetes(return_X_y=True);Xd=StandardScaler().fit_transform(Xd);names=load_diabetes().feature_names
alphas,coefs,_=lasso_path(Xd,yd,n_alphas=200)
f,axs=plt.subplots(1,2,figsize=(10.5,3.6))
for i in range(coefs.shape[0]): axs[0].plot(np.log10(alphas),coefs[i],lw=1.8,color=PAL[i%8])
for i in [2,8,3]: axs[0].text(np.log10(alphas[0])-3.3,coefs[i,-1],names[i],fontsize=7.5,color=PAL[i%8])
axs[0].invert_xaxis();axs[0].set_xlabel("log10(α)  ← stronger regularisation");axs[0].set_ylabel("coefficient");axs[0].set_title("Lasso path: coefficients hit exactly 0")
al=np.logspace(-2,5,100);cs=np.array([Ridge(alpha=a).fit(Xd,yd).coef_ for a in al])
for i in range(cs.shape[1]): axs[1].plot(np.log10(al),cs[:,i],lw=1.8,color=PAL[i%8])
axs[1].invert_xaxis();axs[1].set_xlabel("log10(α)");axs[1].set_title("Ridge path: coefficients shrink smoothly, never 0");save(f,"reg_paths")

# calibration curves
Xb,yb=make_classification(20000,20,n_informative=8,weights=[.7],random_state=2);Xtr,Xte,ytr,yte=train_test_split(Xb,yb,test_size=.5,random_state=0)
f,axs=plt.subplots(1,2,figsize=(10,3.8),gridspec_kw={"width_ratios":[1.3,1]})
axs[0].plot([0,1],[0,1],"--",color=S,label="perfectly calibrated")
for (n,m),c in zip([("Logistic regression",LogisticRegression(max_iter=1000)),("Naive Bayes",GaussianNB()),("Random forest",RandomForestClassifier(200,random_state=0,n_jobs=-1)),("SVM (Platt-free decision fn)",None)],[T,R,A,I]):
    if m is None: continue
    p=m.fit(Xtr,ytr).predict_proba(Xte)[:,1];fr,mp=calibration_curve(yte,p,n_bins=12);axs[0].plot(mp,fr,"o-",ms=4,color=c,label=n)
    if n.startswith("Naive"): axs[1].hist(p,bins=40,color=R,alpha=.6,label="Naive Bayes")
    if n.startswith("Logistic"): axs[1].hist(p,bins=40,color=T,alpha=.6,label="Logistic")
axs[0].set_xlabel("mean predicted probability");axs[0].set_ylabel("observed fraction of positives");axs[0].legend(fontsize=8);axs[0].set_title("Reliability diagram (calibration curve)")
axs[0].text(.55,.12,"below diagonal = over-confident",color=R,fontsize=8);axs[1].set_title("Predicted-probability histograms");axs[1].legend();axs[1].set_xlabel("p̂")
save(f,"calibration")

# threshold tradeoff
m=LogisticRegression(max_iter=1000).fit(Xtr,ytr);p=m.predict_proba(Xte)[:,1];pr,rc,th=precision_recall_curve(yte,p)
f1=2*pr*rc/(pr+rc+1e-9);f,ax=plt.subplots(figsize=(6.8,3.3))
ax.plot(th,pr[:-1],color=T,lw=2,label="precision");ax.plot(th,rc[:-1],color=A,lw=2,label="recall");ax.plot(th,f1[:-1],color=I,lw=2,label="F1")
bt=th[np.argmax(f1[:-1])];ax.axvline(bt,ls="--",color=R);ax.text(bt+.01,.15,f"max-F1 threshold ≈ {bt:.2f}",color=R,weight="bold")
ax.axvline(.5,ls=":",color=S);ax.text(.51,.05,"default 0.5",color=S,fontsize=8);ax.set_xlabel("decision threshold");ax.legend(loc="center left");ax.set_title("Choosing the threshold: precision/recall trade-off")
save(f,"threshold")

# feature importance + shap-like beeswarm (computed via permutation and linear SHAP)
Xbc,ybc=load_breast_cancer(return_X_y=True,as_frame=True);Xtr,Xte,ytr,yte=train_test_split(Xbc,ybc,random_state=0,stratify=ybc)
from sklearn.inspection import permutation_importance
gb=HistGradientBoostingClassifier(random_state=0).fit(Xtr,ytr);pi=permutation_importance(gb,Xte,yte,n_repeats=15,random_state=0,scoring="roc_auc")
o=np.argsort(pi.importances_mean)[-10:]
lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=.3)).fit(Xtr,ytr)
Zs=lr[0].transform(Xte);w=lr[1].coef_[0];phi=Zs*w   # exact SHAP for linear model w/ independent features (log-odds)
oo=np.argsort(np.abs(phi).mean(0))[-10:]
f,axs=plt.subplots(1,2,figsize=(11.5,4.2))
axs[0].barh(range(10),pi.importances_mean[o],xerr=pi.importances_std[o],color=T,ecolor=S);axs[0].set_yticks(range(10));axs[0].set_yticklabels(Xbc.columns[o],fontsize=7.5)
axs[0].set_title("Permutation importance (drop in test ROC-AUC)");axs[0].set_xlabel("mean decrease in AUC")
cmap=LinearSegmentedColormap.from_list("sh",[I,"#E2E8F0",R])
for j,k in enumerate(oo):
    v=Zs[:,k];axs[1].scatter(phi[:,k],j+rng.normal(0,.12,len(v)),c=np.clip(v,-2,2),cmap=cmap,s=7,alpha=.85)
axs[1].set_yticks(range(10));axs[1].set_yticklabels(Xbc.columns[oo],fontsize=7.5);axs[1].axvline(0,color=S,lw=.8)
axs[1].set_xlabel("SHAP value (impact on log-odds of 'benign')");axs[1].set_title("SHAP-style beeswarm (linear model, exact)")
sm=plt.cm.ScalarMappable(cmap=cmap);cb=f.colorbar(sm,ax=axs[1],ticks=[0,1]);cb.ax.set_yticklabels(["low","high"]);cb.set_label("feature value",fontsize=8)
save(f,"importance")

# LR effects
f,axs=plt.subplots(1,2,figsize=(10.5,3.4));ep=np.arange(100)
for lr_,c,lab in [(1e-4,S,"too small (1e-4)"),(1e-3,T,"good (1e-3)"),(1e-2,A,"large (1e-2): fast, noisy"),(1e-1,R,"too large (1e-1): diverges")]:
    if lr_==1e-1: y_=np.minimum(2.3*np.exp(ep/30)*(1+.2*np.sin(ep)),6)
    elif lr_==1e-2: y_=2.3*np.exp(-ep/6)+.35+.12*np.abs(np.sin(ep*1.7))
    elif lr_==1e-3: y_=2.3*np.exp(-ep/18)+.12
    else: y_=2.3*np.exp(-ep/250)+.1
    axs[0].plot(ep,y_,lw=2,color=c,label=lab)
axs[0].set_ylim(0,4);axs[0].set_xlabel("epoch");axs[0].set_ylabel("training loss");axs[0].legend(fontsize=8);axs[0].set_title("Learning-rate effects")
lrs=np.logspace(-6,0,200);loss=2.3-1.9/(1+np.exp(-(np.log10(lrs)+4.2)*2.2))+np.where(lrs>.05,(np.log10(lrs)+1.3)**2*3,0)
axs[1].plot(lrs,loss,color=I,lw=2);axs[1].set_xscale("log");axs[1].set_ylim(0,3);axs[1].axvline(3e-3,ls="--",color=T);axs[1].text(3.5e-3,2.6,"pick ~10× below\nthe minimum",color=T,fontsize=8)
axs[1].set_xlabel("learning rate (exponentially increased)");axs[1].set_ylabel("loss");axs[1].set_title("LR range test (LR finder)");save(f,"lr_effects")

# schedules
st=np.arange(1000);f,ax=plt.subplots(figsize=(6.5,2.8))
ax.plot(st,np.where(st<100,st/100,1)*1e-3,label="constant + warmup",color=S)
ax.plot(st,np.where(st<100,st/100,.5*(1+np.cos(np.pi*(st-100)/900)))*1e-3,label="warmup + cosine",color=T,lw=2)
ax.plot(st,1e-3*0.5**(st//300),label="step decay",color=A);ax.plot(st,np.interp(st,[0,300,1000],[4e-5,1e-3,4e-7]),label="one-cycle",color=I)
ax.set_xlabel("step");ax.set_ylabel("learning rate");ax.legend(fontsize=8);ax.set_title("Learning-rate schedules");save(f,"lr_schedules")

# loss landscapes: convex vs non-convex
Xg,Yg=np.meshgrid(np.linspace(-3,3,200),np.linspace(-3,3,200))
Z1=Xg**2+2*Yg**2;Z2=np.sin(1.5*Xg)*np.cos(1.5*Yg)+.12*(Xg**2+Yg**2)+.3*np.sin(3*Xg)
f=plt.figure(figsize=(11,3.8))
for i,(Z,t) in enumerate([(Z1,"Convex (linear/logistic regression): one global minimum"),(Z2,"Non-convex (neural nets): local minima & saddles")]):
    ax=f.add_subplot(1,3,i+1,projection="3d");ax.plot_surface(Xg,Yg,Z,cmap="viridis",lw=0,alpha=.95,rstride=4,cstride=4);ax.set_title(t,fontsize=8.5);ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
ax=f.add_subplot(1,3,3);cs=ax.contourf(Xg,Yg,Xg**2-Yg**2,30,cmap="RdBu_r");ax.plot(0,0,"k*",ms=12);ax.set_title("Saddle point: ∇L = 0 but not a minimum",fontsize=8.5);ax.set_xticks([]);ax.set_yticks([])
save(f,"landscapes")

# optimizers on Rosenbrock-ish ravine
def grad(p): x,y=p;return np.array([2*x*.1*2/2+0,0])
def f_(p): return .05*p[0]**2+2*p[1]**2
def g_(p): return np.array([.1*p[0],4*p[1]])
paths={}
for name in ["SGD","Momentum","RMSProp","Adam"]:
    p=np.array([-9.,1.6]);v=np.zeros(2);s=np.zeros(2);m=np.zeros(2);P=[p.copy()]
    for t in range(1,61):
        g=g_(p)
        if name=="SGD": p=p-.4*g
        elif name=="Momentum": v=.85*v+g;p=p-.12*v
        elif name=="RMSProp": s=.9*s+.1*g**2;p=p-.3*g/(np.sqrt(s)+1e-8)
        else: m=.9*m+.1*g;s=.999*s+.001*g**2;mh=m/(1-.9**t);sh=s/(1-.999**t);p=p-.5*mh/(np.sqrt(sh)+1e-8)
        P.append(p.copy())
    paths[name]=np.array(P)
Xg,Yg=np.meshgrid(np.linspace(-10,3,300),np.linspace(-2.2,2.2,200));f,ax=plt.subplots(figsize=(9,3.4))
ax.contour(Xg,Yg,f_( (Xg,Yg)),levels=np.logspace(-2,1.2,14),colors="#CBD5E1",linewidths=.8)
for (n,P),c in zip(paths.items(),[S,A,I,T]): ax.plot(P[:,0],P[:,1],"o-",ms=2.5,lw=1.4,color=c,label=n)
ax.plot(0,0,"*",color=R,ms=14);ax.legend(ncol=4,loc="upper center");ax.set_title("Optimizers on an ill-conditioned ravine (60 steps each)");ax.grid(False);save(f,"optimizers")

# softmax temperature
z=np.array([2.,1.,.5,0,-1]);f,axs=plt.subplots(1,3,figsize=(10,2.6),sharey=True)
for ax,Tt in zip(axs,[.5,1,2]):
    p=np.exp(z/Tt)/np.exp(z/Tt).sum();ax.bar(["the","a","my","one","zebra"],p,color=[T,A,I,SKY,R]);ax.set_title(f"temperature T = {Tt}");[ax.text(i,v+.01,f"{v:.2f}",ha="center",fontsize=7.5) for i,v in enumerate(p)]
f.suptitle("softmax(z / T): low T → peaked/greedy, high T → flat/creative",weight="bold",color=N);save(f,"temperature")

# bayes update beta
from scipy import stats
x=np.linspace(0,1,400);f,ax=plt.subplots(figsize=(6.8,3.2))
ax.plot(x,stats.beta(2,2).pdf(x),color=S,lw=2,ls="--",label="prior Beta(2,2)")
ax.plot(x,stats.beta(2+7,2+3).pdf(x),color=A,lw=2,label="after 7/10 successes → Beta(9,5)")
ax.plot(x,stats.beta(2+70,2+30).pdf(x),color=T,lw=2.5,label="after 70/100 → Beta(72,32)")
ax.fill_between(x,stats.beta(72,32).pdf(x),where=(x>stats.beta(72,32).ppf(.025))&(x<stats.beta(72,32).ppf(.975)),color=T,alpha=.2)
ax.set_xlabel("conversion rate θ");ax.set_yticks([]);ax.legend(fontsize=8);ax.set_title("Bayesian updating: posterior sharpens as data arrives");save(f,"bayes_update")

# MLE likelihood curve
data=rng.normal(3,1.5,30);mus=np.linspace(1,5,300);ll=[stats.norm(m,1.5).logpdf(data).sum() for m in mus]
f,ax=plt.subplots(figsize=(6,2.9));ax.plot(mus,ll,color=I,lw=2.2);mh=data.mean();ax.axvline(mh,ls="--",color=R);ax.text(mh+.05,min(ll)+10,f"μ̂ = x̄ = {mh:.2f}",color=R,weight="bold")
ax.set_xlabel("μ");ax.set_ylabel("log-likelihood ℓ(μ)");ax.set_title("MLE: pick the parameter that maximises ℓ");save(f,"mle")

# p-hacking / multiple comparisons
k=np.arange(1,61);f,axs=plt.subplots(1,2,figsize=(10,3.2))
axs[0].plot(k,1-(.95)**k,color=R,lw=2.5,label="no correction");axs[0].plot(k,1-(1-.05/k)**k,color=T,lw=2.5,label="Bonferroni α/m")
axs[0].axhline(.05,ls=":",color=S);axs[0].set_xlabel("number of independent tests m");axs[0].set_ylabel("P(≥1 false positive)");axs[0].legend();axs[0].set_title("Family-wise error explodes")
ps=[]
for _ in range(2000):
    a=rng.normal(0,1,(20,)) ;b=rng.normal(0,1,(20,))
    pv=[stats.ttest_ind(a[:n],b[:n]).pvalue for n in range(5,21)];ps.append(min(pv))
axs[1].hist(ps,bins=40,color=A);axs[1].axvline(.05,color=R,ls="--");axs[1].set_title(f"Peeking after every sample: {np.mean(np.array(ps)<.05):.0%} false positives\n(true effect = 0, nominal α = 5%)",fontsize=9);axs[1].set_xlabel("min p-value over 16 peeks")
save(f,"phacking")

# Simpson's paradox
f,ax=plt.subplots(figsize=(6.5,3.6))
allx=[];ally=[]
for g,(cx,cy),c in [(0,(2,8),T),(1,(5,5),A),(2,(8,2),I)]:
    xs=rng.normal(cx,.7,40);ys=cy+.8*(xs-cx)+rng.normal(0,.4,40);ax.scatter(xs,ys,s=12,color=c,label=f"group {g+1}")
    b=np.polyfit(xs,ys,1);xx_=np.linspace(xs.min(),xs.max(),10);ax.plot(xx_,np.polyval(b,xx_),color=c,lw=2);allx+=list(xs);ally+=list(ys)
b=np.polyfit(allx,ally,1);xx_=np.linspace(0,10,10);ax.plot(xx_,np.polyval(b,xx_),"--",color=R,lw=2.5,label="pooled trend")
ax.legend(fontsize=8);ax.set_title("Simpson's paradox: every group ↑, pooled data ↓");ax.set_xlabel("x (e.g. dose)");ax.set_ylabel("y (e.g. recovery)");save(f,"simpson")

# power curve
from statsmodels.stats.power import NormalIndPower
ns=np.arange(100,40000,200);f,ax=plt.subplots(figsize=(6.5,3))
for d_,c in [(.02,R),(.04,A),(.08,T)]:
    ax.plot(ns,NormalIndPower().power(d_,ns,.05),color=c,lw=2,label=f"Cohen's h = {d_}")
ax.axhline(.8,ls="--",color=S);ax.text(30000,.74,"80% power",color=S);ax.set_xlabel("users per arm");ax.set_ylabel("power");ax.legend();ax.set_title("Power vs sample size (two-proportion test, α=0.05)");save(f,"power")

# bootstrap
d_=rng.lognormal(3,.8,200);bs=[np.median(rng.choice(d_,200)) for _ in range(5000)];lo,hi=np.percentile(bs,[2.5,97.5])
f,ax=plt.subplots(figsize=(6.3,2.8));ax.hist(bs,bins=50,color=I,alpha=.8);ax.axvline(lo,color=R,ls="--");ax.axvline(hi,color=R,ls="--");ax.text(hi+.3,200,f"95% CI\n[{lo:.1f}, {hi:.1f}]",color=R,weight="bold")
ax.set_title("Bootstrap distribution of the median (5,000 resamples)");ax.set_xlabel("median");save(f,"bootstrap")

# Taylor approx
x=np.linspace(-2,2.5,300);f,ax=plt.subplots(figsize=(6,3));ax.plot(x,np.exp(x),color=N,lw=2.5,label="eˣ")
for k_,c in [(1,R),(2,A),(3,T)]:
    import math;ax.plot(x,sum(x**i/math.factorial(i) for i in range(k_+1)),"--",color=c,label=f"order {k_}")
ax.set_ylim(-1,9);ax.legend();ax.set_title("Taylor approximations around 0 (GD uses order 1, Newton order 2)");save(f,"taylor")

# SVD image compression
from sklearn.datasets import load_sample_image
img=load_sample_image("china.jpg").mean(2)[::2,::2];U,Sv,Vt=np.linalg.svd(img,full_matrices=False)
f,axs=plt.subplots(1,4,figsize=(11,2.6))
for ax,k_ in zip(axs,[5,20,60,None]):
    r=img if k_ is None else U[:,:k_]@np.diag(Sv[:k_])@Vt[:k_];ax.imshow(r,cmap="gray");ax.axis("off")
    ax.set_title("original (rank 320)" if k_ is None else f"rank {k_}: {(k_*(img.shape[0]+img.shape[1]+1))/img.size:.0%} of storage",fontsize=8.5)
save(f,"svd")

# Correlation pitfalls: Anscombe-ish
f,axs=plt.subplots(1,4,figsize=(11,2.5))
x=np.linspace(0,10,40);sets=[x+rng.normal(0,2,40),-(x-5)**2/2+10+rng.normal(0,.3,40),np.r_[x[:-1]*0+5+rng.normal(0,1,39),20],rng.normal(5,2,40)]
xs=[x,x,np.r_[x[:-1],30],x]
for ax,xv,yv,t in zip(axs,xs,sets,["linear: r is fine","curved: r≈0 but strong relation","one outlier creates r","no relation"]):
    ax.scatter(xv,yv,s=10,color=T);ax.set_title(f"{t}\nPearson r = {np.corrcoef(xv,yv)[0,1]:.2f}",fontsize=8.5);ax.set_xticks([]);ax.set_yticks([])
save(f,"correlation")
print("charts2 done")
