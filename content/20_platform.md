# Platform Toolbox: Git, Docker, Kubernetes, Cloud, Linux

tagline: The engineering substrate every data and ML engineer touches daily: version control workflows, containers, orchestration, the cloud service map, and the shell commands that save hours.

## Git

![](figures/git.svg)

| Task | Command |
|---|---|
| New branch from main | `git switch -c feat/x origin/main` |
| Stage interactively | `git add -p` |
| Amend last commit (not pushed) | `git commit --amend` |
| Update branch with main | `git fetch && git rebase origin/main` (or `merge`) |
| Squash local commits | `git rebase -i origin/main` |
| Undo a pushed commit safely | `git revert <sha>` |
| Discard local changes to a file | `git restore path` |
| Find the commit that broke it | `git bisect start; git bisect bad; git bisect good <sha>` |
| Who changed this line | `git blame -L 10,20 file.py` |
| Recover "lost" commits | `git reflog` |
| Temporarily shelve work | `git stash push -m "wip"` / `git stash pop` |

!!! pitfall "Pitfalls in data repos"
    Never commit data, models or secrets: use `.gitignore`, DVC/lakeFS/LFS for large files, and a secrets manager. Notebooks create noisy diffs: strip outputs (`nbstripout`) or use jupytext. Rewriting pushed history (`push --force`) on shared branches breaks colleagues; use `--force-with-lease` on your own branches only.

**Workflow:** trunk-based development with short-lived branches, pull requests with CI (lint, type check, tests, data/model checks), conventional commits, and tags for releases. Pre-commit hooks: ruff (lint + format), mypy, nbstripout, detect-secrets.

## Docker

![](figures/docker.svg)

```dockerfile
# multi-stage build for a model API  (reference)
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt
FROM python:3.12-slim
RUN useradd -m app
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
COPY src/ ./src/
COPY models/house_model.joblib ./models/
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

| Practice | Why |
|---|---|
| Slim base image, multi-stage builds | smaller, fewer CVEs |
| Copy dependency files first, code last | layer caching: rebuilds take seconds |
| Pin versions; reference images by digest in production | reproducibility |
| Run as non-root; no secrets in images | security |
| One process per container; log to stdout | orchestration-friendly |
| `.dockerignore` (data, .git, venv) | fast builds, no leaks |
| GPU: `nvidia/cuda` runtime base + matching framework wheels | driver compatibility |

| Command | Purpose |
|---|---|
| `docker build -t api:1.0 .` | build |
| `docker run --rm -p 8000:8000 --env-file .env api:1.0` | run |
| `docker exec -it <id> bash` | shell into a container |
| `docker logs -f <id>` | follow logs |
| `docker compose up -d` | multi-service local stack |
| `docker system prune` | reclaim disk |

## Kubernetes essentials

| Object | Purpose |
|---|---|
| Pod | one or more containers scheduled together |
| Deployment | replicas + rolling updates for stateless services (model APIs) |
| Service / Ingress | stable networking / HTTP routing |
| ConfigMap / Secret | configuration / credentials |
| Job / CronJob | batch training, scheduled scoring |
| HPA | autoscale on CPU, memory or custom metrics (QPS, queue length) |
| Resource requests/limits | scheduling and protection: set both; GPUs via `nvidia.com/gpu` |
| Probes | readiness (ready for traffic) vs liveness (restart if stuck) |

ML on Kubernetes: KServe / Seldon (serving), Kubeflow / Argo Workflows (pipelines), Ray (distributed training/serving), Karpenter / cluster autoscaler (GPU nodes).

## Cloud service map

| Need | AWS | GCP | Azure |
|---|---|---|---|
| Object storage | S3 | Cloud Storage | ADLS Gen2 / Blob |
| Warehouse | Redshift | BigQuery | Synapse / Fabric |
| Lakehouse / Spark | EMR, Glue | Dataproc | Databricks, Fabric |
| Streaming | Kinesis, MSK | Pub/Sub, Dataflow | Event Hubs, Stream Analytics |
| Orchestration | MWAA, Step Functions | Cloud Composer, Workflows | Data Factory |
| ML platform | SageMaker | Vertex AI | Azure ML |
| LLM APIs | Bedrock | Vertex AI (Gemini) | Azure OpenAI |
| Serverless compute | Lambda | Cloud Run / Functions | Functions / Container Apps |
| Containers | ECS, EKS | GKE | AKS |
| Secrets | Secrets Manager | Secret Manager | Key Vault |
| IAM | IAM roles | IAM service accounts | Entra ID, managed identities |

!!! rule "Rule of thumb: cloud cost hygiene"
    Tag everything with owner and project; turn on budgets and alerts; auto-suspend warehouses and notebooks; use spot/preemptible instances for fault-tolerant training; lifecycle old data to cold tiers; watch egress (cross-region and internet) which is often the surprise line item.

## Linux and shell for data work

| Task | Command |
|---|---|
| Peek at a big file | `head -n 5 f.csv`, `wc -l f.csv`, `less -S f.csv` |
| Column counts | `cut -d, -f3 f.csv | sort | uniq -c | sort -rn | head` |
| Search recursively | `rg "pattern" src/` (ripgrep), `grep -rn` |
| Query CSV/Parquet instantly | `duckdb -c "SELECT count(*) FROM 'data/*.parquet'"` |
| JSON | `jq '.items[] | {id, price}' f.json` |
| Disk usage | `df -h`, `du -sh * | sort -h` |
| Processes and resources | `htop`, `nvidia-smi -l 1`, `free -h` |
| Long jobs | `tmux new -s train`, `nohup cmd > log.txt 2>&1 &` |
| Remote copy | `rsync -avzP src/ host:dst/`, `aws s3 sync` |
| Permissions | `chmod +x run.sh`, `chown` |
| Ports | `ss -ltnp`, `lsof -i :8000` |
| Environment | `uv venv && uv pip install -r requirements.txt` |

<div class="card" markdown="1">

## Cheat card: Platform toolbox

<div class="cols" markdown="1">

#### Git
- Short branches, PRs with CI, `revert` not reset on shared history
- `bisect` to find regressions; `reflog` to recover
- No data/secrets in git; nbstripout for notebooks

#### Docker
- Slim base, multi-stage, deps before code
- Pin + digest; non-root; stdout logs
- `.dockerignore`; healthchecks

#### Kubernetes
- Deployment + Service + HPA for model APIs
- Jobs/CronJobs for training & batch scoring
- Requests/limits; readiness vs liveness probes

#### Cloud
- Storage: S3 / GCS / ADLS
- Warehouse: Redshift / BigQuery / Synapse-Fabric
- ML: SageMaker / Vertex / Azure ML
- Tag, budget, auto-suspend, spot, lifecycle, egress

#### Shell
- rg, jq, duckdb, tmux, rsync, htop, nvidia-smi
- `du -sh * | sort -h` when disks fill

</div>
</div>
