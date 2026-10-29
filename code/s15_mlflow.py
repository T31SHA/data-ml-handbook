# HIDE
import warnings, os; warnings.filterwarnings("ignore"); os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///data/mlflow.db"
import logging; logging.disable(logging.WARNING)
# ENDHIDE
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
X, y = load_breast_cancer(return_X_y=True)
mlflow.set_experiment("cancer-baselines")
for C in [0.01, 0.1, 1.0]:
    with mlflow.start_run(run_name=f"logreg_C={C}"):
        mlflow.log_params({"model": "logreg", "C": C})
        auc = cross_val_score(LogisticRegression(C=C, max_iter=5000), X, y, cv=5, scoring="roc_auc").mean()
        mlflow.log_metric("cv_auc", auc); mlflow.set_tag("data_version", "v2026-10-24")
runs = mlflow.search_runs(order_by=["metrics.cv_auc DESC"])
print(runs[["tags.mlflow.runName", "params.C", "metrics.cv_auc"]].round(4).to_string(index=False))
