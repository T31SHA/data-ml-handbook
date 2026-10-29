"""Re-execute every snippet in code/ and refresh code/out/*.txt (the outputs shown in the book)."""
import glob, os, subprocess, sys, textwrap, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(os.path.join(ROOT, "code"))
os.makedirs("data", exist_ok=True); bad = 0
env = dict(os.environ, HANDBOOK_FIG_DIR=os.environ.get("HANDBOOK_FIG_DIR", os.path.join(ROOT, "figures")))
for f in sorted(glob.glob((sys.argv[1] if len(sys.argv) > 1 else "*") + ".py")):
    r = subprocess.run([sys.executable, f], capture_output=True, text=True, timeout=900, env=env)
    lines = r.stdout.rstrip().split("\n")
    if len(lines) > 45: lines = lines[:45] + [f"... ({len(lines)-45} more lines)"]
    if r.returncode == 0:
        open(f"out/{f[:-3]}.txt", "w").write("\n".join("\n".join(textwrap.wrap(l, 108, subsequent_indent="    ", drop_whitespace=False, replace_whitespace=False)) or "" for l in lines))
    else: bad += 1; print(r.stderr[-800:])
    print(f"{f:32s} {'ok' if r.returncode == 0 else 'FAIL'}")
sys.exit(1 if bad else 0)
