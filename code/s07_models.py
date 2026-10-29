# HIDE
import warnings; warnings.filterwarnings("ignore")
# ENDHIDE
import time, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb, xgboost as xgb
X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
models = {"logistic (scaled)": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
          "decision tree d=4": DecisionTreeClassifier(max_depth=4, random_state=0),
          "random forest": RandomForestClassifier(500, n_jobs=-1, random_state=0),
          "hist GBM": HistGradientBoostingClassifier(random_state=0),
          "LightGBM": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1),
          "XGBoost": xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4),
          "SVM RBF (scaled)": make_pipeline(StandardScaler(), SVC(C=3, gamma="scale")),
          "kNN k=7 (scaled)": make_pipeline(StandardScaler(), KNeighborsClassifier(7)),
          "Gaussian NB": GaussianNB()}
rows = []
for name, m in models.items():
    t = time.perf_counter(); s = cross_val_score(m, X, y, cv=cv, scoring="roc_auc")
    rows.append((name, s.mean(), s.std(), time.perf_counter() - t))
print(pd.DataFrame(rows, columns=["model", "ROC-AUC", "std", "sec"]).sort_values("ROC-AUC", ascending=False).round(4).to_string(index=False))
