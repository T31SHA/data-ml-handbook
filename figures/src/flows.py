import subprocess
HEAD='''digraph G { graph [rankdir=TB, nodesep=0.25, ranksep=0.32, fontname="Fira Sans", bgcolor="white", pad=0.15];
node [fontname="Fira Sans", fontsize=12, style="filled,rounded", shape=box, penwidth=1.4, margin="0.12,0.06"];
edge [fontname="Fira Sans", fontsize=10.5, color="#0F172A", fontcolor="#334155", arrowsize=0.7, penwidth=1.1];
'''
def q(n,t): return f'{n} [label="{t}", shape=diamond, style=filled, fillcolor="#FFFBEB", color="#D97706", fontcolor="#0F172A", fontsize=11, margin="0.02,0.02"];\n'
def l(n,t,c="#0D9488"): return f'{n} [label="{t}", fillcolor="{c}", color="{c}", fontcolor="white"];\n'
T="#0D9488";I="#4F46E5";A="#D97706";R="#DB2777";G="#16A34A";S="#475569";SKY="#0284C7";V="#7C3AED"
def build(name,body,lr=True):
    if lr: body='graph [rankdir=LR, ranksep=0.45, nodesep=0.12];\n'+body
    open(f"figures/{name}.dot","w").write(HEAD+body+"}")
    subprocess.run(["dot","-Tsvg",f"figures/{name}.dot","-o",f"figures/{name}.svg"],check=True)

build("flow_model", q("d","Data type?")+q("lab","Labels?")+q("interp","Must be\ninterpretable?")+q("small","< 1k rows?")+q("tl","Labelled\nexamples?")+q("mod","Modality?")+
 l("clu","Clustering: k-means → HDBSCAN/GMM\nAnomalies: Isolation Forest",S)+l("lin","Logistic / linear + splines,\nEBM, shallow tree",I)+l("reg","Regularised linear, RF, SVM\n+ repeated CV",G)+
 l("gbm","Gradient boosting (LightGBM /\nXGBoost / CatBoost) + early stop\n→ Optuna → feature eng.",T)+
 l("llm","LLM zero/few-shot, or\nembeddings + logistic",V)+l("bert","TF-IDF + logistic baseline →\nfine-tune BERT/DeBERTa (LoRA)",R)+
 l("cv","Pretrained CNN / ViT\nfine-tune + augmentation",I)+l("ts","Seasonal-naive → ETS/ARIMA →\nGBM on lags → N-BEATS/TFT",A)+l("rec","Popularity → item-item → MF/ALS\n→ two-tower + GBM ranker",G)+l("rl","Bandits (Thompson) → RL (PPO)",SKY)+
 'd->lab [label="tabular"]; d->tl [label="text"]; d->mod [label="images / audio /\nsignals"]; d->rec [label="user-item\ninteractions"]; d->rl [label="sequential\ndecisions"];\n'
 'lab->clu [label="no"]; lab->interp [label="yes"]; interp->lin [label="yes"]; interp->small [label="no"]; small->reg [label="yes"]; small->gbm [label="no"];\n'
 'tl->llm [label="< 1k"]; tl->bert [label="many"]; mod->cv [label="vision/audio"]; mod->ts [label="time series"];\n')

build("flow_metric", q("t","Task?")+q("bal","Classes\nbalanced?")+q("cal","Need calibrated\nprobabilities?")+q("out","Large errors\nvery costly?")+q("scale","Many scales /\nrelative error?")+
 l("acc","Accuracy, macro-F1, ROC-AUC",T)+l("pr","PR-AUC (AP), recall @ precision ≥ p,\nF-beta",R)+l("ll","Log-loss, Brier score,\nreliability diagram",I)+
 l("rmse","RMSE",A)+l("mae","MAE / median AE (robust)",G)+l("mape","WAPE, MASE, RMSLE\n(MAPE only if y never ≈ 0)",SKY)+l("r2","Also report R² for\nexplained variance",S)+
 l("rank","NDCG@k, MAP@k, recall@k, MRR\nonline: CTR, retention",V)+l("thr","Pick threshold from cost matrix:\nmin C_FP·FP + C_FN·FN",S)+
 't->bal [label="classification"]; t->out [label="regression"]; t->rank [label="ranking /\nrecsys"];\n'
 'bal->acc [label="yes"]; bal->cal [label="no"]; cal->ll [label="yes"]; cal->pr [label="no / ranking"]; pr->thr; acc->thr;\n'
 'out->rmse [label="yes"]; out->scale [label="no"]; scale->mape [label="yes"]; scale->mae [label="no"]; rmse->r2; mae->r2;\n')

build("flow_test", q("y","Outcome\ntype?")+q("g","How many\ngroups?")+q("p","Paired?")+q("cmp","Comparing?")+q("n3","Factors?")+
 l("pt","Paired t-test\n(non-param: Wilcoxon signed-rank)",T)+l("wt","Welch t-test\n(non-param: Mann-Whitney U)",T)+l("an","One-way ANOVA → Tukey HSD\n(non-param: Kruskal-Wallis)",I)+l("an2","Two-way ANOVA /\nlinear model with interactions",I)+
 l("cor","Pearson r (linear) / Spearman ρ\n→ regression with CIs",G)+l("z","Two-proportion z-test\n(A/B conversion)",A)+l("mc","McNemar (paired binary)",A)+l("chi","χ² independence\n(Fisher exact if expected < 5)",R)+l("gof","χ² goodness of fit",R)+l("pois","Poisson / negative-binomial\nregression (counts, rates)",SKY)+
 'y->g [label="numeric"]; y->cmp [label="categorical"]; y->pois [label="counts"];\n'
 'g->p [label="2"]; g->n3 [label="≥ 3"]; g->cor [label="2 numeric\nvariables"]; p->pt [label="yes"]; p->wt [label="no"]; n3->an [label="1"]; n3->an2 [label="2+"];\n'
 'cmp->z [label="2 proportions"]; cmp->mc [label="paired"]; cmp->chi [label="r×c table"]; cmp->gof [label="vs expected"];\n')

build("flow_storage", q("w","Workload?")+q("ap","Access\npattern?")+q("sz","Data size?")+q("lat","Freshness\nneeded?")+
 l("pg","PostgreSQL / MySQL\n(ACID, normalised)",I)+l("kv","DynamoDB / Cassandra /\nRedis (key lookups at scale)",R)+l("doc","MongoDB / Firestore\n(flexible documents)",R)+
 l("duck","DuckDB / Polars on Parquet\n(single node)",T)+l("wh","Warehouse: BigQuery / Snowflake /\nRedshift (SQL, managed)",T)+l("lh","Lakehouse: Spark/Trino +\nIceberg/Delta on object storage",T)+
 l("fl","Kafka + Flink /\nKafka Streams",A)+l("ss","Spark Structured Streaming\n(micro-batch)",A)+l("bt","Batch: Airflow + dbt / Spark\n(hourly / daily)",G)+
 l("sp","Search: OpenSearch · Vectors: pgvector/Qdrant\nGraph: Neo4j · Time series: ClickHouse/Timescale",S)+
 'w->pg [label="transactions"]; pg->ap; ap->kv [label="key-value,\nhuge scale"]; ap->doc [label="flexible\nschema"];\n'
 'w->sz [label="analytics"]; sz->duck [label="< ~100 GB"]; sz->wh [label="TB+, SQL-first"]; sz->lh [label="TB–PB, ML +\nopen formats"];\n'
 'w->lat [label="events"]; lat->fl [label="< 1 s"]; lat->ss [label="seconds–\nminutes"]; lat->bt [label="hours"]; w->sp [label="specialised\nqueries"];\n')

steps=[("Loss at step 0\\n≈ ln(C)?","No → check labels, init, output layer / loss pairing (pass logits!)",R),
("Can it overfit\\none batch?","No → bug: zero_grad, wrong loss, frozen params, detached graph, LR",R),
("Loss NaN / inf?","Yes → lower LR, grad clipping, check inputs, log(0), fp16 → bf16",R),
("Train loss\\nfalls slowly?","LR finder; AdamW 3e-4; normalise inputs; add norm layers; better init",T),
("Train ↓ but\\nval flat / ↑?","Overfitting → augment, dropout, weight decay, early stop, more data",I),
("Both plateau high?","Underfitting → bigger model, train longer, better features, less regularisation",G),
("Val score\\nsuspiciously high?","Leakage → duplicates, future info, preprocessing fit on all data",V)]
body='graph [rankdir=LR, nodesep=0.18, ranksep=0.35];\n'
for i,(qq,a,c) in enumerate(steps):
    body+=q(f"q{i}",qq)+l(f"a{i}",a,c)+f'q{i}->a{i};\n'
body+="{rank=same; "+" ".join(f"q{i}" for i in range(7))+"}\n"
body+=" ".join(f"q{i}->q{i+1} [style=dashed, color=\"#94A3B8\"];" for i in range(6))+"\n"
# vertical arrangement: use rankdir LR with rank=same for questions -> they stack vertically
build("flow_debug",body)
print("flows ok")
