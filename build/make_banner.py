"""Render the README hero banner (assets/banner.png) in the handbook palette."""
import os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N="#0F172A"; PAL=["#0D9488","#4F46E5","#D97706","#DB2777","#16A34A","#0284C7","#7C3AED"]
f=plt.figure(figsize=(12.8,4),dpi=110); ax=f.add_axes([0,0,1,1]); ax.set_xlim(0,12.8); ax.set_ylim(0,4); ax.axis("off")
ax.add_patch(plt.Rectangle((0,0),12.8,4,color=N))
rng=np.random.default_rng(7)
# right side: a small neural net + loss curve motif
L=[3,5,5,2]; xs=np.linspace(8.4,12.0,len(L)); pts=[]
for x,n in zip(xs,L): pts.append([(x,y) for y in np.linspace(2-0.33*(n-1),2+0.33*(n-1),n)])
for a,b in zip(pts,pts[1:]):
    for p in a:
        for q in b: ax.plot([p[0],q[0]],[p[1],q[1]],color="#334155",lw=.7,zorder=1)
for i,layer in enumerate(pts):
    for p in layer: ax.add_patch(plt.Circle(p,.13,color=PAL[i%len(PAL)],zorder=2))
t=np.linspace(0,1,80); ax.plot(7.6+t*4.6,0.45+0.9*np.exp(-5*t)+0.03*rng.standard_normal(80),color="#0D9488",lw=2)
for i,c in enumerate(PAL): ax.add_patch(plt.Rectangle((0.6+i*0.55,0.55),0.45,0.08,color=c))
ax.text(.6,3.25,"THE ELITE FIELD GUIDE",color="#5EEAD4",fontsize=13,weight="bold")
ax.text(.6,2.25,"Data · ML · AI ·\nData Engineering Handbook",color="white",fontsize=30,weight="bold",va="center",linespacing=1.05)
ax.text(.6,1.0,"From raw CSV to production: math, stats, classical ML, deep learning, LLMs, pipelines",color="#CBD5E1",fontsize=11.5)
ax.text(12.5,0.2,"SHARAHBIL MOHAMED ABDI · DATA/ML ENGINEER",color="#94A3B8",fontsize=8.5,ha="right",weight="bold")
f.savefig(os.path.join(ROOT,"assets","banner.png")); print("banner ok")
