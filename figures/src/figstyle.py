import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, Polygon
N="#0F172A";T="#0D9488";A="#D97706";R="#DB2777";I="#4F46E5";S="#64748B";G="#16A34A";SKY="#0284C7";V="#7C3AED"
PAL=[T,A,I,R,G,SKY,V,S]
plt.rcParams.update({"font.family":["Fira Sans","DejaVu Sans"],"font.size":9.5,"axes.spines.top":False,"axes.spines.right":False,
 "axes.edgecolor":"#94A3B8","axes.titleweight":"bold","axes.titlesize":10.5,"axes.titlecolor":N,"axes.labelcolor":"#334155",
 "xtick.color":"#475569","ytick.color":"#475569","axes.prop_cycle":plt.cycler(color=PAL),"axes.grid":True,"grid.color":"#E2E8F0","grid.linewidth":.6,
 "legend.frameon":False,"svg.fonttype":"path","figure.dpi":110,"mathtext.fontset":"dejavusans"})
import os
O=os.environ.get("HANDBOOK_FIG_DIR",os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))+"/"
RASTER={"proj_eda","boundaries","landscapes","gradient_descent","proj_eval"}  # dense plots -> PNG keeps the repo small
def save(f,n):
    f.tight_layout(); f.savefig(O+n+(".png" if n in RASTER else ".svg"),bbox_inches="tight",dpi=220); plt.close(f)
def box(ax,x,y,w,h,txt,c,fc="white",fs=8.5,tc=None,bold=True,ls="-"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.08",ec=c,fc=fc,lw=1.8,ls=ls))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=tc or N,weight="bold" if bold else "normal")
def fbox(ax,x,y,w,h,txt,c,fs=8.5):  # filled
    box(ax,x,y,w,h,txt,c,fc=c,fs=fs,tc="white")
def diamond(ax,x,y,w,h,txt,c,fs=8):
    ax.add_patch(Polygon([(x,y+h/2),(x+w/2,y+h),(x+w,y+h/2),(x+w/2,y)],closed=True,ec=c,fc="#FFFBEB",lw=1.8))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=N,weight="bold")
def arr(ax,x1,y1,x2,y2,c=S,lab=None,lw=1.5,ls="-",rad=0):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="-|>",color=c,lw=lw,ls=ls,connectionstyle=f"arc3,rad={rad}"))
    if lab: ax.text((x1+x2)/2,(y1+y2)/2,lab,fontsize=7.5,color=c,ha="center",va="center",bbox=dict(fc="white",ec="none",pad=.6))
def canvas(w,h,xl=None,yl=None):
    f,ax=plt.subplots(figsize=(w,h)); ax.set_xlim(0,xl or w); ax.set_ylim(0,yl or h); ax.axis("off"); return f,ax
def callout(ax,xy,xytext,txt,c=R,fs=8):
    ax.annotate(txt,xy=xy,xytext=xytext,fontsize=fs,color=c,weight="bold",arrowprops=dict(arrowstyle="->",color=c,lw=1.2),bbox=dict(boxstyle="round,pad=0.3",fc="white",ec=c,lw=.8))
