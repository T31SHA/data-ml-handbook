"""Shared helpers: expand @@code/@@out include directives used in content/*.md."""
import re, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIDE = re.compile(r"^# *HIDE.*?# *ENDHIDE\n", re.S | re.M)

def code_body(path):
    return HIDE.sub("", open(os.path.join(ROOT, path)).read()).rstrip()

def expand(src, out_label="Output"):
    def code(m):
        path, lang = m.group(1), m.group(2) or ("sql" if m.group(1).endswith(".sql") else "python")
        return f"```{lang}\n{code_body(path)}\n```"
    def out(m):
        p = os.path.join(ROOT, m.group(1))
        t = open(p).read().rstrip() if os.path.exists(p) else "(output missing)"
        return f"**{out_label}**\n\n```text\n{t}\n```"
    src = re.sub(r"^@@code (\S+)(?: (\w+))?@@$", code, src, flags=re.M)
    return re.sub(r"^@@out (\S+)@@$", out, src, flags=re.M)

def snippets(md_path):
    return re.findall(r"^@@code (code/\S+\.py)@@$", open(md_path).read(), flags=re.M)
