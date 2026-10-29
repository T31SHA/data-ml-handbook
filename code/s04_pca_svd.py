import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
X = load_iris().data; Xc = X - X.mean(0)
evals, evecs = np.linalg.eigh(np.cov(Xc.T)); evals, evecs = evals[::-1], evecs[:, ::-1]
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
pca = PCA().fit(X)
print("eig of covariance  :", evals.round(4))
print("S² / (n−1) from SVD:", (S**2 / (len(X) - 1)).round(4))
print("sklearn explained  :", pca.explained_variance_.round(4))
print("explained ratio    :", pca.explained_variance_ratio_.round(3), "-> 2 PCs keep", pca.explained_variance_ratio_[:2].sum().round(3))
print("same directions (|cos|):", np.abs((evecs[:, 0] * Vt[0]).sum()).round(6))
