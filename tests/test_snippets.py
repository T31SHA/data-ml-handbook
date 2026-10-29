"""Every code snippet printed in the handbook must run cleanly (in sorted order: data-prep scripts come first)."""
import glob, os, subprocess, sys, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "code")
SNIPS = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CODE, "*.py")))
SLOW = {"proj04_tune_eval.py"}
LIBS = {"housing_feats.py"}  # helper modules imported by other snippets

@pytest.mark.parametrize("snippet", SNIPS)
def test_snippet_runs(snippet, tmp_path):
    if snippet in SLOW and os.environ.get("SKIP_SLOW"): pytest.skip("slow snippet (unset SKIP_SLOW to run)")
    os.makedirs(os.path.join(CODE, "data"), exist_ok=True)
    env = dict(os.environ, HANDBOOK_FIG_DIR=str(tmp_path), MPLBACKEND="Agg", KERAS_BACKEND=os.environ.get("KERAS_BACKEND", "tensorflow"))
    r = subprocess.run([sys.executable, snippet], cwd=CODE, capture_output=True, text=True, timeout=900, env=env)
    assert r.returncode == 0, f"{snippet} failed:\n{r.stderr[-3000:]}"
    if snippet not in LIBS: assert r.stdout.strip(), f"{snippet} printed nothing"

def test_every_include_exists():
    import re
    missing = []
    for md in glob.glob(os.path.join(ROOT, "content", "*.md")):
        for path in re.findall(r"^@@(?:code|out) ([^@\s]+)", open(md).read(), re.M):
            if not os.path.exists(os.path.join(ROOT, path)): missing.append((os.path.basename(md), path))
        for img in re.findall(r"\]\((figures/[^)\s]+)\)", open(md).read()):
            if not os.path.exists(os.path.join(ROOT, img)): missing.append((os.path.basename(md), img))
    assert not missing, missing
