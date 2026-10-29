import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, log_loss, brier_score_loss, matthews_corrcoef
y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]); p = np.array([.9, .8, .6, .3, .7, .4, .2, .2, .1, .05])
yhat = (p >= .5).astype(int); tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"precision={precision_score(y, yhat):.3f} (= {tp}/{tp+fp})  recall={recall_score(y, yhat):.3f} (= {tp}/{tp+fn})  F1={f1_score(y, yhat):.3f}")
print(f"ROC-AUC={roc_auc_score(y, p):.3f}  log-loss={log_loss(y, p):.3f}  Brier={brier_score_loss(y, p):.3f}  MCC={matthews_corrcoef(y, yhat):.3f}")
pairs = [(i, j) for i in np.where(y == 1)[0] for j in np.where(y == 0)[0]]
print("AUC as P(score_pos > score_neg) =", np.mean([p[i] > p[j] for i, j in pairs]).round(3))
