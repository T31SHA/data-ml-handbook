# Linear Algebra

tagline: Vectors are data points, matrices are transformations, and eigen/singular decompositions reveal the directions that matter.

## Vectors and the dot product

$$\mathbf{x}\cdot\mathbf{y} = \mathbf{x}^\top\mathbf{y} = \sum_{i=1}^{d} x_i y_i = \lVert\mathbf{x}\rVert\,\lVert\mathbf{y}\rVert\cos\theta$$

!!! intuition "Intuition"
    The dot product measures how much two vectors point the same way, scaled by their lengths. A neuron computes $\mathbf{w}^\top\mathbf{x}$ ("how much does the input look like my template?"), attention computes $\mathbf{q}^\top\mathbf{k}$ ("how relevant is this token?"), and linear regression predicts $\mathbf{w}^\top\mathbf{x}$.

!!! example "Worked example"
    $\mathbf{x} = (1, 2, 2)$, $\mathbf{y} = (2, 0, 1)$: $\mathbf{x}^\top\mathbf{y} = 2 + 0 + 2 = 4$; $\lVert\mathbf{x}\rVert = 3$, $\lVert\mathbf{y}\rVert = \sqrt{5}$; $\cos\theta = 4/(3\sqrt5) \approx 0.596$, so $\theta \approx 53°$.

| Quantity | Formula | Use |
|---|---|---|
| L2 norm | $\lVert\mathbf{x}\rVert_2 = \sqrt{\sum x_i^2}$ | Euclidean length, ridge penalty |
| L1 norm | $\lVert\mathbf{x}\rVert_1 = \sum \lvert x_i\rvert$ | Manhattan distance, lasso penalty (sparsity) |
| L∞ norm | $\max_i \lvert x_i\rvert$ | worst-case deviation, adversarial bounds |
| Cosine similarity | $\dfrac{\mathbf{x}^\top\mathbf{y}}{\lVert\mathbf{x}\rVert\lVert\mathbf{y}\rVert}$ | embeddings (direction, not magnitude) |
| Euclidean distance | $\lVert\mathbf{x}-\mathbf{y}\rVert_2$ | kNN, k-means |
| Projection of $\mathbf{y}$ on $\mathbf{x}$ | $\dfrac{\mathbf{x}^\top\mathbf{y}}{\mathbf{x}^\top\mathbf{x}}\,\mathbf{x}$ | least squares, Gram–Schmidt |

For unit vectors, $\lVert\mathbf{x}-\mathbf{y}\rVert^2 = 2 - 2\cos\theta$, so ranking by cosine similarity equals ranking by Euclidean distance once embeddings are normalised.

## Matrices as transformations

$$(A\mathbf{x})_i = \sum_j A_{ij}x_j, \qquad (AB)_{ij} = \sum_k A_{ik}B_{kj}, \qquad (m\times n)(n\times p) \to (m\times p)$$

| Concept | Definition | Meaning |
|---|---|---|
| Transpose | $(A^\top)_{ij} = A_{ji}$, $(AB)^\top = B^\top A^\top$ | swap rows and columns |
| Inverse | $AA^{-1} = I$ (square, $\det A \ne 0$) | undo the transformation |
| Determinant | $\det A = \prod_i \lambda_i$ | volume scale factor; 0 means space is squashed (singular) |
| Rank | number of independent columns | true dimensionality of the information |
| Trace | $\operatorname{tr}A = \sum_i A_{ii} = \sum_i \lambda_i$ | sum of eigenvalues |
| Orthogonal | $Q^\top Q = I$ | rotation/reflection; preserves lengths and angles |
| Symmetric | $A = A^\top$ | real eigenvalues, orthogonal eigenvectors |
| Positive definite | $\mathbf{x}^\top A\mathbf{x} > 0\ \forall\mathbf{x}\ne 0$ | all $\lambda_i > 0$; bowl-shaped quadratic; valid covariance |
| Null space | $\{\mathbf{x}: A\mathbf{x} = 0\}$ | directions the transform destroys (non-identifiable parameters) |

Four views of $A\mathbf{x}$: a weighted **combination of columns**; a stack of **dot products with rows**; a **transformation** of space; a **linear model** applied to features.

## Eigendecomposition

$$A\mathbf{v} = \lambda\mathbf{v}, \qquad A = Q\Lambda Q^\top \ \text{(symmetric } A\text{)}$$

![](figures/eigen.svg)

!!! example "Worked example"
    $A = \begin{pmatrix}2 & 1\\1 & 2\end{pmatrix}$. $\det(A - \lambda I) = (2-\lambda)^2 - 1 = 0 \Rightarrow \lambda \in \{1, 3\}$. For $\lambda = 3$: $\mathbf{v} = (1,1)/\sqrt2$; for $\lambda = 1$: $\mathbf{v} = (1,-1)/\sqrt2$. Check: $\operatorname{tr}A = 4 = 1 + 3$ and $\det A = 3 = 1\cdot 3$.

**Where it appears:** PCA (eigenvectors of the covariance), PageRank and Markov chains (eigenvalue 1), spectral clustering (graph Laplacian), stability of RNNs (spectral radius > 1 explodes, < 1 vanishes), Hessian curvature (min/max eigenvalues set the condition number).

## Singular value decomposition (SVD)

$$X = U\Sigma V^\top, \qquad U \in \mathbb{R}^{n\times r},\ \Sigma = \operatorname{diag}(\sigma_1 \ge \dots \ge \sigma_r > 0),\ V \in \mathbb{R}^{d\times r}$$

!!! intuition "Intuition"
    Every matrix is *rotate → stretch → rotate*. Keeping the top-$k$ singular values gives the best rank-$k$ approximation in Frobenius norm (Eckart–Young): $X_k = U_k\Sigma_kV_k^\top$, error $= \sqrt{\sum_{i>k}\sigma_i^2}$.

![](figures/svd.svg)

**Uses:** PCA (right singular vectors of centred $X$), latent semantic analysis, recommender matrix factorisation, low-rank compression and LoRA, pseudo-inverse $X^+ = V\Sigma^{-1}U^\top$, numerically stable least squares, condition number $\kappa = \sigma_{\max}/\sigma_{\min}$.

## PCA = eigen-decomposition = SVD (verified)

@@code code/s04_pca_svd.py@@
@@out code/out/s04_pca_svd.txt@@

![](figures/pca.svg)

!!! derivation "Derivation: PCA maximises projected variance"
    Centre the data. The variance of projections onto a unit vector $\mathbf{w}$ is $\mathbf{w}^\top C\mathbf{w}$ with $C = \frac{1}{n-1}X^\top X$. Maximise subject to $\mathbf{w}^\top\mathbf{w} = 1$ with a Lagrangian: $\nabla_\mathbf{w}\big[\mathbf{w}^\top C\mathbf{w} - \lambda(\mathbf{w}^\top\mathbf{w} - 1)\big] = 2C\mathbf{w} - 2\lambda\mathbf{w} = 0 \Rightarrow C\mathbf{w} = \lambda\mathbf{w}$. The projected variance equals $\lambda$, so the best direction is the top eigenvector; subsequent components are the next eigenvectors (orthogonal by symmetry).

## Least squares and the normal equation

!!! derivation "Derivation: OLS normal equation"
    Minimise $\mathcal{L}(\mathbf{w}) = \lVert X\mathbf{w} - \mathbf{y}\rVert^2 = \mathbf{w}^\top X^\top X\mathbf{w} - 2\mathbf{y}^\top X\mathbf{w} + \mathbf{y}^\top\mathbf{y}$. Gradient: $\nabla\mathcal{L} = 2X^\top X\mathbf{w} - 2X^\top\mathbf{y}$. Setting it to zero gives the **normal equation** $X^\top X\,\hat{\mathbf{w}} = X^\top\mathbf{y}$, so $\hat{\mathbf{w}} = (X^\top X)^{-1}X^\top\mathbf{y}$ when $X^\top X$ is invertible. The Hessian $2X^\top X$ is PSD, so this is a global minimum. Geometrically, $X\hat{\mathbf{w}}$ is the orthogonal projection of $\mathbf{y}$ onto the column space of $X$: the residual is orthogonal to every feature, $X^\top(\mathbf{y} - X\hat{\mathbf{w}}) = 0$. Ridge adds $\lambda I$: $\hat{\mathbf{w}} = (X^\top X + \lambda I)^{-1}X^\top\mathbf{y}$, which is always invertible.

@@code code/s04_ols.py@@
@@out code/out/s04_ols.txt@@

!!! pitfall "Pitfall: never form an explicit inverse"
    `np.linalg.inv(X.T @ X) @ X.T @ y` squares the condition number and amplifies rounding errors. Use `np.linalg.lstsq` (SVD/QR), `np.linalg.solve`, or `scipy.linalg.cho_solve` for PD systems. Multicollinearity (κ > 10⁴–10⁶) also makes coefficients unstable: regularise.

| Complexity | Cost |
|---|---|
| matrix–vector $(n\times d)$ | $O(nd)$ |
| matrix–matrix $(n\times d)(d\times p)$ | $O(ndp)$ |
| solve / inverse / determinant $(d\times d)$ | $O(d^3)$ |
| full SVD $(n\times d)$ | $O(nd\min(n,d))$; truncated/randomised SVD $O(ndk)$ |
| OLS via normal equation | $O(nd^2 + d^3)$ |

!!! interview "Interview question: why does multicollinearity hurt linear regression but not prediction accuracy much?"
    Correlated features make $X^\top X$ nearly singular, so many coefficient combinations fit almost equally well: individual coefficients get huge variance and flip signs, and their p-values are meaningless. Predictions in the region covered by the data stay stable because the *combination* is well determined. Ridge, PCA or dropping redundant features stabilise the coefficients.
