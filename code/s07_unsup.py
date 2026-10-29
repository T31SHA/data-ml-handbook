# HIDE
import warnings; warnings.filterwarnings("ignore")
# ENDHIDE
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
X, y = make_blobs(1000, centers=4, cluster_std=1.2, random_state=0)
for k in range(2, 7):
    km = KMeans(k, n_init=10, random_state=0).fit(X)
    print(f"k={k} inertia={km.inertia_:8.0f} silhouette={silhouette_score(X, km.labels_):.3f}")
gm = [GaussianMixture(k, random_state=0).fit(X).bic(X) for k in range(2, 7)]
print("GMM BIC best k =", 2 + int(np.argmin(gm)))
Xm, ym = make_moons(600, noise=0.06, random_state=0)
for name, m in [("k-means", KMeans(2, n_init=10, random_state=0)), ("agglomerative-single", AgglomerativeClustering(2, linkage="single")),
                ("DBSCAN eps=.2", DBSCAN(eps=0.2, min_samples=5)), ("HDBSCAN", HDBSCAN(min_cluster_size=20))]:
    print(f"moons  {name:22s} ARI={adjusted_rand_score(ym, m.fit_predict(Xm)):.3f}")
