# HIDE
import duckdb, glob, os
con = duckdb.connect("data/shop.duckdb")
for f in sorted(glob.glob("sql/*.sql")):
    stmts = [s for s in open(f).read().split(";") if s.strip() and not all(l.strip().startswith("--") or not l.strip() for l in s.splitlines())]
    res = None
    for s in stmts: res = con.execute(s)
    df = res.df()
    open("out/" + os.path.basename(f)[:-4] + ".txt", "w").write(df.to_string(index=False, max_colwidth=40))
    print(os.path.basename(f), "rows:", len(df))
# ENDHIDE
