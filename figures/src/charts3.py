from figstyle import *
# ---------- backprop computational graph ----------
f,ax=canvas(11,3.6)
nodes={"x":(0.3,1.6,"x"),"W1":(0.3,2.7,"W₁, b₁"),"z1":(2.2,1.9,"z₁ = W₁x + b₁"),"h":(4.3,1.9,"h = ReLU(z₁)"),"W2":(4.3,3.0,"W₂, b₂"),"z2":(6.4,1.9,"z₂ = W₂h + b₂"),"p":(8.3,1.9,"p = softmax(z₂)"),"L":(10.0,1.9,"L = −log p_y")}
for k,(x,y,t) in nodes.items():
    w=1.0 if k in("x","W1","W2") else 1.65
    fbox(ax,x-0.05,y-0.3,w,0.6,t,{ "x":S,"W1":I,"W2":I,"L":R}.get(k,T),fs=8)
for a,b in [("x","z1"),("W1","z1"),("z1","h"),("h","z2"),("W2","z2"),("z2","p"),("p","L")]:
    xa,ya,_=nodes[a];xb,yb,_=nodes[b];wa=1.0 if a in("x","W1","W2") else 1.65
    arr(ax,xa+wa-0.05,ya,xb-0.05,yb,c=N)
for t,(x,y) in [("∂L/∂z₂ = p − onehot(y)",(7.6,0.9)),("∂L/∂W₂ = (p−y) hᵀ",(5.6,0.45)),("∂L/∂h = W₂ᵀ(p−y)",(5.0,0.9)),("∂L/∂z₁ = ∂L/∂h ⊙ 1[z₁>0]",(3.2,0.45)),("∂L/∂W₁ = ∂L/∂z₁ · xᵀ",(1.6,0.9))]:
    ax.text(x,y,t,fontsize=8.5,color=R,weight="bold",ha="center")
ax.annotate("",xy=(0.6,0.15),xytext=(10.6,0.15),arrowprops=dict(arrowstyle="-|>",color=R,lw=2));ax.text(10.6,0.25,"backward pass (chain rule)",color=R,ha="right",fontsize=8)
ax.annotate("",xy=(10.6,3.5),xytext=(0.6,3.5),arrowprops=dict(arrowstyle="-|>",color=T,lw=2));ax.text(0.6,3.25,"forward pass (compute & cache activations)",color=T,fontsize=8)
save(f,"backprop_graph")

# ---------- transformer full ----------
f,ax=canvas(10,6.4)
def stack(x0,title,layers,c):
    ax.add_patch(FancyBboxPatch((x0,1.2),3.4,4.1,boxstyle="round,pad=0.05,rounding_size=0.15",ec=c,fc="#F8FAFC",lw=1.5,ls="--"))
    ax.text(x0+1.7,5.45,title,ha="center",weight="bold",color=c)
    y=1.45
    for t,cc in layers: fbox(ax,x0+0.25,y,2.9,0.5,t,cc,fs=8);y+=0.68
stack(0.3,"ENCODER  × N",[("Multi-head self-attention",T),("Add & LayerNorm",S),("Feed-forward (MLP, 4× width)",A),("Add & LayerNorm",S)],T)
stack(5.6,"DECODER  × N",[("Masked (causal) self-attention",R),("Add & LayerNorm",S),("Cross-attention (Q from decoder,\nK,V from encoder)",I),("Add & LayerNorm",S),("Feed-forward + Add & Norm",A)],I)
fbox(ax,0.55,0.2,2.9,0.6,"Token embeddings + positions",N,fs=8);fbox(ax,5.85,0.2,2.9,0.6,"Shifted output tokens + positions",N,fs=8)
arr(ax,2,0.8,2,1.4,c=N);arr(ax,7.3,0.8,7.3,1.4,c=N);arr(ax,3.7,3.6,5.85,3.45,c=I,lab="memory (K,V)")
fbox(ax,5.85,5.75,2.9,0.5,"Linear → softmax over vocab",G,fs=8);arr(ax,7.3,5.6,7.3,5.75,c=N)
ax.text(5,6.3,"Encoder-only: BERT (understanding) · Decoder-only: GPT/Llama (generation) · Enc-dec: T5 (seq2seq)",ha="center",fontsize=8.5,color=N,style="italic")
save(f,"transformer_full")

# ---------- attention mechanics ----------
f,ax=canvas(11,3.2)
fbox(ax,0.2,1.3,1.1,0.7,"X\n(n×d)",S)
for i,(t,c) in enumerate([("Q = XW_Q",T),("K = XW_K",A),("V = XW_V",I)]): fbox(ax,2,2.3-i*0.95,1.4,0.6,t,c,fs=8);arr(ax,1.3,1.65,2,2.6-i*0.95,c=S)
fbox(ax,4.2,1.8,1.9,0.7,"S = QKᵀ / √dₖ\n(n×n scores)",R,fs=8);arr(ax,3.4,2.6,4.2,2.2,c=N);arr(ax,3.4,1.65,4.2,2.0,c=N)
fbox(ax,6.6,1.8,1.6,0.7,"+ mask\nsoftmax(rows)",V,fs=8);arr(ax,6.1,2.15,6.6,2.15,c=N)
fbox(ax,8.7,1.3,1.9,0.7,"Output = A·V\n(n×d_v)",G,fs=8);arr(ax,8.2,2.1,8.7,1.8,c=N);arr(ax,3.4,0.65,8.7,1.45,c=I,rad=.15)
ax.text(5.5,0.25,"Multi-head: h parallel heads with d_k = d/h, concatenated then projected by W_O.  Cost O(n²·d) time, O(n²) memory.",ha="center",fontsize=8,color=N)
save(f,"attention_mech")

# ---------- KV cache ----------
f,ax=canvas(10,3)
toks=["The","cat","sat","on","the"]
for i,t in enumerate(toks):
    fbox(ax,0.3+i*1.15,1.9,1,0.5,t,S if i<4 else T,fs=8)
    ax.add_patch(plt.Rectangle((0.3+i*1.15,1.1),1,0.6,fc="#CCFBF1" if i<4 else "#FDE68A",ec=T));ax.text(0.8+i*1.15,1.4,f"K{i},V{i}",ha="center",fontsize=7.5,color=N)
ax.text(0.3,0.75,"cached from previous steps  →  only the new token's K,V are computed",fontsize=8,color=T)
fbox(ax,6.6,1.5,1.5,0.7,"attend: q₄ · [K₀..K₄]",I,fs=8);arr(ax,6.0,1.8,6.6,1.85,c=N);fbox(ax,8.5,1.5,1.2,0.7,"next token\n'mat'",G,fs=8);arr(ax,8.1,1.85,8.5,1.85,c=N)
ax.text(5,2.75,"KV-cache memory ≈ 2 · layers · kv_heads · head_dim · seq_len · batch · bytes",ha="center",fontsize=8.5,weight="bold",color=R)
ax.text(5,0.3,"Prefill (prompt, parallel, compute-bound) → Decode (1 token/step, memory-bandwidth-bound)",ha="center",fontsize=8,color=N)
save(f,"kv_cache")

# ---------- LoRA ----------
f,ax=canvas(8,3.2)
ax.add_patch(plt.Rectangle((0.5,0.6),2.2,2.2,fc="#E2E8F0",ec=S));ax.text(1.6,1.7,"W₀\n(d×k)\nfrozen",ha="center",va="center",weight="bold",color=N)
ax.text(3.1,1.7,"+",fontsize=22,ha="center",va="center",color=N)
ax.add_patch(plt.Rectangle((3.6,0.6),0.45,2.2,fc="#99F6E4",ec=T));ax.text(3.82,1.7,"B\nd×r",ha="center",va="center",fontsize=8,weight="bold")
ax.add_patch(plt.Rectangle((4.2,2.35),2.2,0.45,fc="#FDE68A",ec=A));ax.text(5.3,2.57,"A  (r×k)",ha="center",va="center",fontsize=8,weight="bold")
ax.text(5.3,1.5,"h = W₀x + (α/r)·BAx\nr ≪ min(d,k), e.g. 8–64\nB init 0 ⇒ starts as W₀",ha="center",fontsize=8.5,color=N)
ax.text(4,0.2,"Trainable params: r(d+k) instead of dk  (d=k=4096, r=16 → 0.8%)",ha="center",fontsize=8.5,color=R,weight="bold")
save(f,"lora")

# ---------- tokenization ----------
f,ax=canvas(10,1.6)
pieces=[("Token",T),("ization",A),(" is",I),(" surprisingly",R),(" un",G),("intuitive",SKY),("!",S)];x=0.2
for p,c in pieces:
    w=0.16*len(p)+0.35;fbox(ax,x,0.7,w,0.5,p.replace(" ","␣"),c,fs=8.5);x+=w+0.08
ax.text(0.2,0.25,"BPE splits text into frequent sub-words: 7 tokens for 5 words.  Rule of thumb (English): 1 token ≈ 4 chars ≈ ¾ word.",fontsize=8.3,color=N)
save(f,"tokens")

# ---------- medallion ----------
f,ax=canvas(11,3.3)
fbox(ax,0.2,1.2,1.6,1.1,"Sources\nOLTP · APIs\nevents · files",S,fs=8)
for i,(t,sub,c) in enumerate([("BRONZE","raw, append-only\nas ingested, full history",A),("SILVER","cleaned, deduped,\nconformed, typed, joined",S),("GOLD","business aggregates,\nstar schemas, features",A)]):
    x=2.4+i*2.6;cc=["#B45309","#64748B","#CA8A04"][i]
    fbox(ax,x,1.0,2.1,1.5,f"{t}\n\n{sub}",cc,fs=8);arr(ax,x-0.6,1.75,x,1.75,c=N)
arr(ax,1.8,1.75,2.4,1.75,c=N)
fbox(ax,10.0,2.0,0.9,0.6,"BI",G,fs=8);fbox(ax,10.0,1.0,0.9,0.6,"ML",R,fs=8);arr(ax,9.6,1.9,10,2.3,c=N);arr(ax,9.6,1.6,10,1.3,c=N)
ax.text(5.5,0.45,"Each hop: schema enforcement · data-quality tests · idempotent MERGE · lineage.   Tables in Delta/Iceberg on object storage.",ha="center",fontsize=8,color=N)
ax.text(5.5,3.05,"Medallion (multi-hop) architecture",ha="center",weight="bold",color=N,fontsize=10.5)
save(f,"medallion")

# ---------- CDC ----------
f,ax=canvas(11,2.8)
fbox(ax,0.2,1.0,1.7,1.0,"Postgres/MySQL\n(OLTP)",I,fs=8);fbox(ax,2.4,1.0,1.5,1.0,"WAL / binlog",S,fs=8);fbox(ax,4.4,1.0,1.6,1.0,"Debezium\n(Kafka Connect)",T,fs=8)
fbox(ax,6.5,1.0,1.6,1.0,"Kafka topic\nper table",N,fs=8);fbox(ax,8.6,1.6,2.2,0.7,"Stream processor / sink",A,fs=8);fbox(ax,8.6,0.6,2.2,0.7,"Lakehouse MERGE (upsert)",G,fs=8)
for a,b in [(1.9,2.4),(3.9,4.4),(6.0,6.5)]: arr(ax,a,1.5,b,1.5,c=N)
arr(ax,8.1,1.6,8.6,1.95,c=N);arr(ax,8.1,1.4,8.6,0.95,c=N)
ax.text(5.5,2.5,"Change Data Capture: stream every INSERT/UPDATE/DELETE with before/after images, op type, LSN & timestamp",ha="center",fontsize=8.5,weight="bold",color=N)
ax.text(5.5,0.2,"No load on source queries · low latency · captures deletes · requires ordering per key + idempotent upserts",ha="center",fontsize=8,color=S)
save(f,"cdc")

# ---------- feature store ----------
f,ax=canvas(11,3.6)
fbox(ax,0.2,2.3,1.7,0.8,"Batch sources\n(warehouse)",S,fs=8);fbox(ax,0.2,0.9,1.7,0.8,"Streaming\n(Kafka)",S,fs=8)
fbox(ax,2.5,1.6,2.0,0.9,"Feature pipelines\n(definitions as code)",T,fs=8)
fbox(ax,5.2,2.4,2.1,0.8,"OFFLINE store\n(Parquet/Delta, history)",I,fs=8);fbox(ax,5.2,0.8,2.1,0.8,"ONLINE store\n(Redis/DynamoDB, latest)",A,fs=8)
fbox(ax,8.0,2.4,2.8,0.8,"Training: point-in-time-correct\njoin → no leakage",I,fs=8);fbox(ax,8.0,0.8,2.8,0.8,"Serving: low-latency lookup\nby entity key (<10 ms)",A,fs=8)
for (a,b,c_,d) in [(1.9,2.7,2.5,2.2),(1.9,1.3,2.5,1.9),(4.5,2.2,5.2,2.8),(4.5,1.9,5.2,1.2),(7.3,2.8,8,2.8),(7.3,1.2,8,1.2)]: arr(ax,a,b,c_,d,c=N)
arr(ax,6.25,2.4,6.25,1.6,c=R,lab="materialise");ax.text(5.5,3.45,"Feature store: same feature logic for training and serving (kills training–serving skew)",ha="center",weight="bold",color=N,fontsize=9.5)
save(f,"feature_store")

# ---------- serving architecture ----------
f,ax=canvas(11,4.2)
fbox(ax,0.2,1.8,1.3,0.8,"Client\napp",S,fs=8);fbox(ax,2.0,1.8,1.5,0.8,"API gateway\nauth · rate limit",N,fs=8)
fbox(ax,4.0,1.8,1.9,0.8,"Model service\n(FastAPI/Triton)\nautoscaled pods",T,fs=8)
fbox(ax,4.0,3.2,1.9,0.6,"Feature store (online)",A,fs=8);fbox(ax,4.0,0.4,1.9,0.6,"Model registry\n(versioned artefacts)",I,fs=8)
fbox(ax,6.6,2.9,1.9,0.7,"Prediction log\n(Kafka → lake)",V,fs=8);fbox(ax,6.6,1.0,1.9,0.7,"Shadow / canary\nmodel v2",R,fs=8)
fbox(ax,9.0,2.9,1.8,0.7,"Monitoring\ndrift · latency · KPIs",G,fs=8);fbox(ax,9.0,1.0,1.8,0.7,"Retraining\npipeline",SKY,fs=8)
for a in [(1.5,2.2,2.0,2.2),(3.5,2.2,4.0,2.2),(4.95,2.6,4.95,3.2),(4.95,1.0,4.95,1.8),(5.9,2.4,6.6,3.1),(5.9,2.0,6.6,1.4),(8.5,3.25,9.0,3.25),(9.9,2.9,9.9,1.7)]: arr(ax,*a,c=N)
arr(ax,9.0,1.2,5.9,0.7,c=SKY,rad=-.2,lab="register new version")
ax.text(5.5,4.05,"Online model-serving architecture",ha="center",weight="bold",color=N,fontsize=10.5)
save(f,"serving_arch")

# ---------- batch vs online inference patterns ----------
f,ax=canvas(11,2.6)
ax.text(0.2,2.3,"Batch",weight="bold",color=T);[fbox(ax,0.2+i*2.2,1.5,1.9,0.6,t,T,fs=8) for i,t in enumerate(["Scheduler (Airflow)","Read features (lake)","Score in bulk (Spark)","Write predictions table","App reads table"])]
ax.text(0.2,1.05,"Online",weight="bold",color=A);[fbox(ax,0.2+i*2.2,0.25,1.9,0.6,t,A,fs=8) for i,t in enumerate(["Request arrives","Fetch online features","Model inference <100ms","Return + log","Async feedback"])]
for i in range(4): arr(ax,2.1+i*2.2,1.8,2.4+i*2.2,1.8,c=N);arr(ax,2.1+i*2.2,0.55,2.4+i*2.2,0.55,c=N)
save(f,"inference_patterns")

# ---------- Spark DAG ----------
f,ax=canvas(11,3.6)
for s,(x0,w,t,c) in enumerate([(0.2,3.4,"Stage 1 (narrow ops, pipelined)",T),(4.0,3.0,"Stage 2",I),(7.4,3.4,"Stage 3",A)]):
    ax.add_patch(FancyBboxPatch((x0,0.5),w,2.5,boxstyle="round,pad=0.03",ec=c,fc="#F8FAFC",lw=1.4,ls="--"));ax.text(x0+w/2,3.15,t,ha="center",weight="bold",color=c,fontsize=8.5)
for i,t in enumerate(["read parquet","filter","withColumn"]): fbox(ax,0.4+i*1.08,1.5,0.95,0.6,t,T,fs=7.5)
fbox(ax,4.2,1.5,1.2,0.6,"groupBy\n(partial agg)",I,fs=7.5);fbox(ax,5.7,1.5,1.1,0.6,"agg",I,fs=7.5)
fbox(ax,7.6,1.5,1.2,0.6,"join",A,fs=7.5);fbox(ax,9.1,1.5,1.4,0.6,"write\n(action)",R,fs=7.5)
arr(ax,1.35,1.8,1.48,1.8,c=N);arr(ax,2.43,1.8,2.56,1.8,c=N)
arr(ax,3.5,1.8,4.2,1.8,c=R,lab="SHUFFLE");arr(ax,6.8,1.8,7.6,1.8,c=R,lab="SHUFFLE");arr(ax,8.8,1.8,9.1,1.8,c=N);arr(ax,5.4,1.8,5.7,1.8,c=N)
ax.text(5.5,0.15,"Job (triggered by action) → stages (split at shuffle boundaries) → tasks (one per partition)",ha="center",fontsize=8.5,color=N)
save(f,"spark_dag")

# ---------- causal DAGs ----------
f,axs=plt.subplots(1,4,figsize=(11,2.6))
def dag(ax,nodes,edges,title,note):
    ax.axis("off");ax.set_xlim(0,3);ax.set_ylim(0,2.4)
    for n,(x,y,c) in nodes.items(): ax.add_patch(Circle((x,y),.28,fc=c,ec=N,lw=1));ax.text(x,y,n,ha="center",va="center",color="white",weight="bold",fontsize=9)
    for a,b in edges:
        (x1,y1,_),(x2,y2,_)=nodes[a],nodes[b];ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="-|>",color=N,shrinkA=12,shrinkB=12,lw=1.4))
    ax.set_title(title,fontsize=9.5);ax.text(1.5,0.05,note,ha="center",fontsize=7.6,color=R)
dag(axs[0],{"X":(.5,.8,T),"Y":(2.5,.8,A),"Z":(1.5,1.9,I)},[("Z","X"),("Z","Y"),("X","Y")],"Confounder (fork)","adjust for Z")
dag(axs[1],{"X":(.5,.8,T),"Y":(2.5,.8,A),"M":(1.5,1.9,I)},[("X","M"),("M","Y")],"Mediator (chain)","don't adjust if you want total effect")
dag(axs[2],{"X":(.5,1.9,T),"Y":(2.5,1.9,A),"C":(1.5,.8,R)},[("X","C"),("Y","C")],"Collider","NEVER adjust: opens a spurious path")
dag(axs[3],{"Z":(.4,1.6,V),"X":(1.4,1.0,T),"Y":(2.6,1.0,A),"U":(2.0,2.0,S)},[("Z","X"),("X","Y"),("U","X"),("U","Y")],"Instrument","Z→X only; U unobserved")
save(f,"causal_dags")

# ---------- RAG eval triangle ----------
f,ax=canvas(6.5,3.3)
pts={"Question":(3.25,2.9),"Context":(0.8,0.5),"Answer":(5.7,0.5)}
for n,(x,y) in pts.items(): fbox(ax,x-0.75,y-0.25,1.5,0.5,n,{"Question":I,"Context":T,"Answer":A}[n],fs=8.5)
arr(ax,2.55,2.65,1.2,0.8,c=N,lab="context relevance\n/ precision@k");arr(ax,1.6,0.5,4.9,0.5,c=N,lab="faithfulness / groundedness");arr(ax,5.3,0.8,3.95,2.65,c=N,lab="answer relevance")
ax.text(3.25,1.55,"RAG triad",ha="center",weight="bold",color=R,fontsize=10);save(f,"rag_triad")

# ---------- lambda vs kappa ----------
f,ax=canvas(11,2.8)
ax.text(0.2,2.5,"Lambda",weight="bold",color=T);fbox(ax,0.2,1.4,1.4,0.6,"events",S,fs=8);fbox(ax,2.1,1.85,2.0,0.5,"batch layer (Spark)",T,fs=8);fbox(ax,2.1,1.05,2.0,0.5,"speed layer (Flink)",A,fs=8);fbox(ax,4.6,1.4,1.4,0.6,"serving\nmerge",I,fs=8)
arr(ax,1.6,1.8,2.1,2.1,c=N);arr(ax,1.6,1.6,2.1,1.3,c=N);arr(ax,4.1,2.1,4.6,1.8,c=N);arr(ax,4.1,1.3,4.6,1.6,c=N)
ax.text(6.6,2.5,"Kappa",weight="bold",color=A);fbox(ax,6.6,1.4,1.3,0.6,"log\n(Kafka)",S,fs=8);fbox(ax,8.3,1.4,1.4,0.6,"stream\nprocessor",A,fs=8);fbox(ax,10.0,1.4,0.9,0.6,"views",I,fs=8)
arr(ax,7.9,1.7,8.3,1.7,c=N);arr(ax,9.7,1.7,10,1.7,c=N);ax.text(8.75,0.9,"reprocess = replay the log",ha="center",fontsize=8,color=R)
ax.text(3.1,0.5,"two code paths (complex), exact batch recompute",ha="center",fontsize=8,color=S);ax.text(8.7,0.5,"one code path; needs long retention",ha="center",fontsize=8,color=S)
save(f,"lambda_kappa")
print("charts3 done")
