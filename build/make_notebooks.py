"""Generate one Colab-ready notebook per section from the runnable snippets in code/."""
import glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__)); from common import ROOT, code_body, snippets
os.chdir(ROOT); os.makedirs("notebooks", exist_ok=True)
REPO = "T31SHA/data-ml-handbook"
def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(True)}
def py(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.splitlines(True)}
SETUP = """# Setup: works in Colab or locally from the repo root
import os, subprocess, sys
if not os.path.exists("code") and not os.path.exists("../code"):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/%s.git"], check=True)
    os.chdir("data-ml-handbook")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=False)
if os.path.exists("../code"): os.chdir("..")
os.chdir("code"); os.makedirs("data", exist_ok=True)
for prep in ["proj01_make_csv.py", "s02_load.py"]:          # shared datasets used by many snippets
    if not os.path.exists("data/housing.parquet"): subprocess.run([sys.executable, prep], check=True)
sys.path.insert(0, "../figures/src")""" % REPO
n = 0
for p in sorted(glob.glob("content/*.md")):
    snips = snippets(p)
    if not snips: continue
    name = os.path.basename(p)[:-3]
    title = re.search(r"^# (.*)$", open(p).read(), re.M).group(1)
    cells = [md(f"# {title}\n\n[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
                f"(https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{name}.ipynb)\n\n"
                f"Runnable examples from **Data/ML Handbook** — section `{name}`. "
                f"Read the explanations in the [docs](https://t31sha.github.io/data-ml-handbook/handbook/{name}/)."),
             py(SETUP)]
    for s in snips:
        cells += [md(f"### `{s}`"), py(code_body(s))]
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
          "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(f"notebooks/{name}.ipynb", "w"), indent=1); n += 1
print("notebooks:", n)
