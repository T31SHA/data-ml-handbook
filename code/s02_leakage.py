# HIDE
import warnings; warnings.filterwarnings("ignore")
# ENDHIDE
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 5000)); y = rng.integers(0, 2, 200)        # pure noise: true accuracy = 50%
X_leaky = SelectKBest(f_classif, k=20).fit_transform(X, y)              # WRONG: selection sees all labels
print("leaky CV accuracy :", cross_val_score(LogisticRegression(), X_leaky, y, cv=5).mean().round(3))
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression())  # RIGHT: selection inside each fold
print("honest CV accuracy:", cross_val_score(pipe, X, y, cv=5).mean().round(3))
