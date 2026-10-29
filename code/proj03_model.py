# HIDE
import warnings; warnings.filterwarnings("ignore")
# ENDHIDE
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer, make_column_selector as sel
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import lightgbm as lgb

df = pd.read_csv("data/housing_raw.csv").drop_duplicates()                 # 1. dedupe
df["region"] = df["region"].str.strip().str.replace("socal", "SoCal").str.replace("norcal", "NorCal").str.replace("central", "Central")
df["region"] = df["region"].str.replace("-inland", "-Inland").str.replace("-coast", "-Coast")
from housing_feats import add_feats   # lives in a module: pickled pipelines must import it at load time
y = df.pop("MedHouseVal"); X = df
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)   # 3. split BEFORE fitting anything

num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())])
pre = ColumnTransformer([("num", num, sel(dtype_include="number")),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), ["region"])])
make = lambda model: Pipeline([("feats", FunctionTransformer(add_feats)), ("pre", pre), ("model", model)])
models = {"Baseline (mean)": make(DummyRegressor()),
          "Ridge": make(RidgeCV(alphas=np.logspace(-3, 3, 13))),
          "Random forest": make(RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=0)),
          "HistGradientBoosting": make(HistGradientBoostingRegressor(max_iter=500, learning_rate=0.08, random_state=0)),
          "LightGBM": make(lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                             subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0))}
cv = KFold(5, shuffle=True, random_state=0)
rows = []
for name, m in models.items():
    r = cross_validate(m, X_tr, y_tr, cv=cv, n_jobs=1, scoring=("neg_root_mean_squared_error", "r2"))
    rows.append((name, -r["test_neg_root_mean_squared_error"].mean(), r["test_neg_root_mean_squared_error"].std(), r["test_r2"].mean()))
print(pd.DataFrame(rows, columns=["model", "CV RMSE", "± std", "CV R²"]).round(3).to_string(index=False))
