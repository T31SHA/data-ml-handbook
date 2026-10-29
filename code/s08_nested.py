# HIDE
import warnings; warnings.filterwarnings("ignore")
# ENDHIDE
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
X, y = load_breast_cancer(return_X_y=True)
inner, outer = StratifiedKFold(3, shuffle=True, random_state=1), StratifiedKFold(5, shuffle=True, random_state=2)
search = GridSearchCV(make_pipeline(StandardScaler(), SVC()), {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.001]}, cv=inner)
search.fit(X, y)
print(f"best inner-CV score (optimistic): {search.best_score_:.4f}  params={search.best_params_}")
nested = cross_val_score(search, X, y, cv=outer)
print(f"nested CV estimate (unbiased):    {nested.mean():.4f} ± {nested.std():.4f}")
