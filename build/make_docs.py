"""Generate the MkDocs site sources (docs/handbook/*, docs/figures) from content/*.md."""
import glob, os, re, shutil, sys
sys.path.insert(0, os.path.dirname(__file__)); from common import ROOT, expand
os.chdir(ROOT)
out = "docs/handbook"; shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
shutil.rmtree("docs/figures", ignore_errors=True); shutil.copytree("figures", "docs/figures", ignore=shutil.ignore_patterns("src", "*.dot"))
for p in sorted(glob.glob("content/*.md")):
    s = expand(open(p).read(), "Output (executed in CI)")
    s = re.sub(r"^tagline: (.*)$", r"> *\1*", s, count=1, flags=re.M)
    s = s.replace("](figures/", "](../figures/").replace('src="figures/', 'src="../figures/')
    nb = os.path.basename(p)[:-3]
    if os.path.exists(f"notebooks/{nb}.ipynb"):
        badge = (f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
                 f"(https://colab.research.google.com/github/T31SHA/data-ml-handbook/blob/main/notebooks/{nb}.ipynb)")
        s = re.sub(r"^(# .*)$", r"\1\n\n" + badge.replace("\\", "\\\\"), s, count=1, flags=re.M)
    open(os.path.join(out, os.path.basename(p)), "w").write(s)
print("docs generated:", len(os.listdir(out)), "pages")
