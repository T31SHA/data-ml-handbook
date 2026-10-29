# HIDE
import warnings; warnings.filterwarnings("ignore")
# ENDHIDE
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, f1_score, accuracy_score
X, y = make_classification(50_000, 20, n_informative=6, weights=[0.99], flip_y=0.002, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.3, random_state=0)
X_tr2, X_val, y_tr2, y_val = train_test_split(X_tr, y_tr, stratify=y_tr, test_size=0.3, random_state=0)
print(f"positives: {y.mean():.2%}   'always 0' accuracy = {accuracy_score(y_te, np.zeros_like(y_te)):.4f}  <- useless")
for cw in [None, "balanced"]:
    m = LogisticRegression(class_weight=cw, max_iter=2000).fit(X_tr2, y_tr2)
    pv, pt = m.predict_proba(X_val)[:, 1], m.predict_proba(X_te)[:, 1]
    P, R, T = precision_recall_curve(y_val, pv); f1 = 2 * P * R / (P + R + 1e-12); t = T[np.argmax(f1[:-1])]
    print(f"class_weight={str(cw):8s} ROC-AUC={roc_auc_score(y_te, pt):.3f} PR-AUC={average_precision_score(y_te, pt):.3f} "
          f"F1@0.5={f1_score(y_te, pt > .5):.3f}  F1@tuned({t:.2f})={f1_score(y_te, pt > t):.3f}")
