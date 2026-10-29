# HIDE
import warnings; warnings.filterwarnings("ignore")
# ENDHIDE
import pandas as pd, numpy as np
import pandera.pandas as pa
schema = pa.DataFrameSchema({
    "order_id": pa.Column(int, unique=True),
    "amount": pa.Column(float, [pa.Check.ge(0), pa.Check.le(1e6)]),
    "status": pa.Column(str, pa.Check.isin(["new", "paid", "refund"])),
    "email": pa.Column(str, pa.Check.str_matches(r"^[^@\s]+@[^@\s]+\.\w+$"), nullable=True)}, strict=True)
bad = pd.DataFrame({"order_id": [1, 2, 2], "amount": [10.0, -5.0, 30.0], "status": ["paid", "PAID", "new"],
                    "email": ["a@x.com", "nope", None]})
try: schema.validate(bad, lazy=True)
except pa.errors.SchemaErrors as e:
    print(e.failure_cases[["column", "check", "failure_case"]].to_string(index=False))
