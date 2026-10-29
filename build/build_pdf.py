import markdown, re, glob, html, json, os, subprocess, hashlib, sys
from weasyprint import HTML
from pygments.formatters import HtmlFormatter
"""Build the handbook PDF: content/*.md -> HTML (math pre-rendered to SVG by MathJax/Node) -> WeasyPrint PDF.
Usage: python build/build_pdf.py [out.pdf]   (run `npm ci --prefix build/mathjax` once)"""
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
OUT=sys.argv[1] if len(sys.argv)>1 else "dist/Data-ML-Handbook.pdf"
os.makedirs(os.path.dirname(OUT) or ".",exist_ok=True)
B="build/"; MATH="build/.cache/math"; os.makedirs(MATH,exist_ok=True)
COLORS=["#0D9488","#4F46E5","#D97706","#DB2777","#16A34A","#0284C7","#7C3AED","#EA580C","#0F766E","#9333EA","#B45309","#BE123C","#15803D","#1D4ED8","#C2410C","#475569","#0891B2","#A21CAF"]
secs=sorted(glob.glob("content/*.md"))

# ---------- math ----------
mcache_f=MATH+"/cache.json"; mcache=json.load(open(mcache_f)) if os.path.exists(mcache_f) else {}
pending={}
def mkey(tex,disp): return hashlib.md5((("D" if disp else "I")+tex).encode()).hexdigest()[:12]
def mathref(tex,disp):
    k=mkey(tex,disp)
    if k not in mcache: pending[k]={"id":k,"tex":tex,"display":disp}
    return k
def include(src):
    def code(m):
        path=m.group(1);lang=m.group(2) or ("sql" if path.endswith(".sql") else "python")
        body=open(path).read().rstrip()
        body=re.sub(r"^# *HIDE.*?# *ENDHIDE\n","",body,flags=re.S|re.M)
        return f"```{lang}\n{body}\n```"
    def out(m):
        p=m.group(1); t=open(p).read().rstrip() if os.path.exists(p) else "(output missing)"
        return '<div class="outlbl">Output (executed on the build box)</div>\n\n```text\n'+t+"\n```"
    src=re.sub(r"^@@code (\S+)(?: (\w+))?@@$",code,src,flags=re.M)
    src=re.sub(r"^@@out (\S+)@@$",out,src,flags=re.M)
    return src
def protect_code(src,store):
    def rep(m): store.append(m.group(0)); return f"\n§CODE{len(store)-1}§\n" if m.group(0).startswith("```") else f"§CODE{len(store)-1}§"
    src=re.sub(r"^```.*?^```[ \t]*$",rep,src,flags=re.S|re.M)
    src=re.sub(r"`[^`\n]+`",rep,src)
    return src
def math_extract(src,mstore):
    src=src.replace(r"\$","§DOLLAR§")
    def d(m): mstore.append((m.group(1).strip(),True)); return f"\n\nMJXD{len(mstore)-1}X\n\n"
    def i(m): mstore.append((m.group(1).strip(),False)); return f"MJXI{len(mstore)-1}X"
    src=re.sub(r"\$\$(.+?)\$\$",d,src,flags=re.S)
    src=re.sub(r"\$([^\$\n]+?)\$",i,src)
    return src
def svg_img(k,disp):
    svg=mcache.get(k,"")
    mw=re.search(r'width="([\d.]+)ex" height="([\d.]+)ex"',svg)
    if not mw: return '<span class="merr">[math]</span>'
    mv=re.search(r'vertical-align: ([-\d.]+)ex',svg); va=mv.group(1) if mv else "0"
    w,h=mw.groups(); p=f"{MATH}/{k}.svg"
    if not os.path.exists(p): open(p,"w").write(svg.replace("currentColor","#0F172A"))
    if not os.path.exists(p[:-4]+"_w.svg"): open(p[:-4]+"_w.svg","w").write(svg.replace("currentColor","#FFFFFF"))
    sc=0.93 if disp else 0.97
    st=f"width:{float(w)*sc:.3f}ex;height:{float(h)*sc:.3f}ex;vertical-align:{float(va)*sc:.3f}ex"
    return f'<img class="{"md" if disp else "mi"}" src="{p}" style="{st}">'

# ---------- markdown ----------
md=markdown.Markdown(extensions=["tables","fenced_code","codehilite","attr_list","md_in_html","sane_lists","admonition","def_list"],
    extension_configs={"codehilite":{"guess_lang":False,"css_class":"hl"}})
docs=[];open_all=''
for p in secs:
    open_all+=open(p).read()+'\n'
    src=include(open(p).read()); store=[]; mstore=[]
    src=protect_code(src,store); src=math_extract(src,mstore)
    for i_,c in enumerate(store): src=src.replace(f"§CODE{i_}§",c)
    for t,dp in mstore: mathref(t,dp)
    docs.append((p,src,mstore))
if pending:
    json.dump(list(pending.values()),open(MATH+"/in.json","w"))
    r=subprocess.run(["node",B+"mathjax/render.js",MATH+"/in.json",MATH+"/out.json"],capture_output=True,text=True)
    print(r.stdout.strip(), r.stderr.strip()[:2000])
    if r.returncode: sys.exit("MathJax render failed - did you run `npm ci --prefix build/mathjax`?")
    mcache.update(json.load(open(MATH+"/out.json"))); json.dump(mcache,open(mcache_f,"w"))

body="";toc="";sections_html=[]
for i,(p,src,mstore) in enumerate(docs,1):
    md.reset(); h=md.convert(src); c=COLORS[(i-1)%len(COLORS)]
    def mrep(m):
        idx=int(m.group(2)); t,dp=mstore[idx]; img=svg_img(mkey(t,dp),dp)
        return f'<div class="mathd">{img}</div>' if dp else img
    h=re.sub(r"<p>MJX(D)(\d+)X</p>",mrep,h); h=re.sub(r"MJX([DI])(\d+)X",mrep,h)
    h=h.replace("§DOLLAR§","$")
    if False: h=re.sub(r"<th>(.*?)</th>",lambda m:"<th>"+re.sub(r'src="(build/\.cache/math/[0-9a-f]+)\.svg"',r'src="\1_w.svg"',m.group(1))+"</th>",h,flags=re.S)
    title=re.search(r"<h1>(.*?)</h1>",h).group(1); sid=f"s{i}"
    tag=""
    mt=re.search(r"<p>tagline: (.*?)</p>",h)
    if mt: tag=mt.group(1); h=h.replace(mt.group(0),"",1)
    subs=[]; cnt=[0]
    def rep(m):
        cnt[0]+=1; hid=f"{sid}_{cnt[0]}"; subs.append((hid,re.sub("<.*?>","",m.group(1)))); return f'<h2 id="{hid}">{m.group(1)}</h2>'
    h=re.sub(r"<h2>(.*?)</h2>",rep,h)
    h3c=[0]
    def rep3(m): h3c[0]+=1; return f'<h3 id="{sid}_h{h3c[0]}">{m.group(1)}</h3>'
    h=re.sub(r"<h3>(.*?)</h3>",rep3,h)
    h=re.sub(r"<p><img (?!class=\"m)",'<p class="fig"><img ',h)
    mini="".join(f'<li>{t}</li>' for _,t in subs if not t.lower().startswith("cheat card"))
    plain=re.sub("<.*?>","",title).replace("&amp;","&")
    SHORT={"Statistics":"Statistics & Causal Inference","Probability":"Probability & Information Theory","MLOps":"MLOps & ML System Design"}
    short=plain if len(plain)<=40 else re.split(r":|,| & ",plain)[0]
    short=SHORT.get(short,short) if len(plain)>40 else short
    opener=f'''<div class="opener"><div class="secname">{html.escape(short)}</div><div class="onum">{i-1:02d}</div><h1 id="{sid}">{title}</h1>
<div class="otag">{tag}</div><ul class="omini">{mini}</ul></div>'''
    h=h.replace(f"<h1>{title}</h1>",opener,1)
    sections_html.append((sid,title,h,subs))
    body+=f'<section style="--c:{c}">{h}</section>'
    toc+=f'<li class="t1"><a href="#{sid}"><span class="dot" style="background:{c}"></span>{i-1:02d}&nbsp;&nbsp;{title}</a></li>'
    toc+="".join(f'<li class="t2"><a href="#{a}">{t}</a></li>' for a,t in subs)

# ---------- index ----------
idx_html=""
if os.path.exists(B+"index_terms.txt"):
    entries=[]
    blocks=[]
    for sid,title,h,subs in sections_html:
        parts=re.split(r'(<h[23] id="[^"]+">)',h)
        cur=sid; last=(sid==sections_html[-1][0])
        for part in parts:
            m=re.match(r'<h[23] id="([^"]+)">',part)
            if m: cur=m.group(1); continue
            txt=re.sub(r"<[^>]+>"," ",part); head=txt.split("\n",1)[0]
            blocks.append((cur,txt,head,last))
    for line in open(B+"index_terms.txt"):
        line=line.strip()
        if not line or line.startswith("#"): continue
        term,_,pat=line.partition("|"); pat=pat or re.escape(term)
        rx=re.compile(r"(?<![\w-])(?:"+pat+r")(?![\w-])",re.I)
        cand=[]
        for o,(a,txt,head,last) in enumerate(blocks):
            if rx.search(txt): cand.append(((0 if rx.search(head) else 1),(1 if last else 0),-len(rx.findall(txt)),o,a))
        hits=[]
        for c in sorted(cand):
            if c[4] not in hits: hits.append(c[4])
        hits=hits[:4]
        order={a:o for o,(a,*_) in enumerate(blocks)}
        hits.sort(key=lambda a:order[a])
        if hits: entries.append((term,hits))
    entries.sort(key=lambda e:e[0].lower())
    cur="";li=""
    for term,hits in entries:
        L=term[0].upper()
        if L!=cur: cur=L; li+=f'<li class="il">{L if L.isalpha() else "#"}</li>'
        refs=", ".join(f'<a class="pref" href="#{a}"></a>' for a in hits)
        li+=f'<li class="ie">{html.escape(term)} <span class="refs">{refs}</span></li>'
    idx_html=f'<section class="index" style="--c:#0F172A"><div class="ihead"><h1 id="index">Index</h1><p>Page references point to the subsection where each term is defined or used.</p></div><ul class="ilist">{li}</ul></section>'
    toc+='<li class="t1"><a href="#index"><span class="dot" style="background:#0F172A"></span>Index</a></li>'

css=HtmlFormatter(style="friendly_grayscale").get_style_defs(".hl")
tpl=open(B+"style.css").read()
cover=open(B+"cover.html").read()
doc=f"""<html lang="en"><head><meta charset="utf-8"><style>{tpl}\n{css}</style></head><body>
{cover}
<div class="toc"><h1 class="toch">Contents</h1><ul>{toc}</ul></div>
{body}{idx_html}</body></html>"""
open("build/.cache/handbook.html","w").write(doc)
nfig=len(set(re.findall(r'src="(figures/[^"]+)"',doc)))
nsn=len(set(re.findall(r"^@@out (\S+)@@",open_all,flags=re.M)))
nform=len(set(x.replace("_w","") for x in re.findall(r'src="(build/\.cache/math/[^"]+)"',doc)))
def render(pages):
    d=doc.replace("__PAGES__",str(pages)).replace("__FIGS__",str(nfig)).replace("__SNIPS__",str(nsn)).replace("__FORM__",str(nform))
    open("build/.cache/handbook.html","w").write(d); return HTML("build/.cache/handbook.html",base_url=ROOT).render()
r=render("—"); n=len(r.pages); r=render(n); r.write_pdf(OUT)
print("pages",len(r.pages),"figs",nfig,"snips",nsn,"formulas",nform)
print("built",OUT)
