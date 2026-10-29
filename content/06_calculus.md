# Calculus & Optimisation

tagline: Limits, derivatives, integrals, gradients, Jacobians and Hessians, and the optimisers that turn them into learning.

## Limits and continuity

$$\lim_{x\to a} f(x) = L \iff \forall\varepsilon>0\ \exists\delta>0:\ 0<\lvert x-a\rvert<\delta \Rightarrow \lvert f(x)-L\rvert<\varepsilon$$

| Limit | Value | Where it shows up |
|---|---|---|
| $\lim_{x\to0}\frac{\sin x}{x}$ | 1 | small-angle approximations |
| $\lim_{n\to\infty}\left(1+\frac{1}{n}\right)^n$ | $e$ | continuous compounding, exponential growth |
| $\lim_{x\to0}\frac{e^x-1}{x}$ | 1 | why `expm1`/`log1p` exist for precision |
| $\lim_{x\to\infty}\sigma(x)$, $\lim_{x\to-\infty}\sigma(x)$ | 1, 0 | sigmoid saturation → vanishing gradients |
| $\lim_{T\to0}\operatorname{softmax}(\mathbf{z}/T)$ | one-hot argmax | greedy decoding |

**L'Hôpital:** for $0/0$ or $\infty/\infty$ forms, $\lim f/g = \lim f'/g'$. **Continuity** at $a$: $\lim_{x\to a} f(x) = f(a)$. Differentiable implies continuous, not conversely (ReLU at 0 is continuous but not differentiable; frameworks use a subgradient of 0).

## Derivatives

$$f'(x) = \lim_{h\to0}\frac{f(x+h)-f(x)}{h}$$

![](figures/calculus.svg)

!!! intuition "Intuition"
    The derivative is the local slope: "if I nudge $x$ a little, how much does $f$ change?" In ML, $\partial\mathcal{L}/\partial w$ says how the loss responds to each weight, and gradient descent moves every weight a small step downhill.

| $f(x)$ | $f'(x)$ | Rule | Formula |
|---|---|---|---|
| $x^n$ | $nx^{n-1}$ | sum | $(f+g)' = f'+g'$ |
| $e^x$ | $e^x$ | product | $(fg)' = f'g + fg'$ |
| $\ln x$ | $1/x$ | quotient | $(f/g)' = (f'g - fg')/g^2$ |
| $a^x$ | $a^x\ln a$ | **chain** | $\frac{d}{dx}f(g(x)) = f'(g(x))\,g'(x)$ |
| $\sigma(x)$ | $\sigma(x)(1-\sigma(x))$ | inverse | $(f^{-1})'(y) = 1/f'(x)$ |
| $\tanh x$ | $1-\tanh^2 x$ | log-derivative | $(\ln f)' = f'/f$ |
| $\operatorname{ReLU}(x)$ | $\mathbb{1}[x>0]$ | softplus $\ln(1+e^x)$ | $\sigma(x)$ |

!!! derivation "Derivation: the sigmoid derivative"
    $\sigma(x) = (1+e^{-x})^{-1}$. Chain rule: $\sigma'(x) = -(1+e^{-x})^{-2}\cdot(-e^{-x}) = \frac{e^{-x}}{(1+e^{-x})^2} = \sigma(x)\cdot\frac{e^{-x}}{1+e^{-x}} = \sigma(x)(1-\sigma(x))$. Its maximum is $0.25$ at $x = 0$, so stacking $L$ sigmoid layers multiplies gradients by at most $0.25^L$: the **vanishing-gradient** problem.

## Taylor series: the local polynomial view

$$f(x+h) = f(x) + f'(x)h + \tfrac12 f''(x)h^2 + \tfrac16 f'''(x)h^3 + \cdots$$

![](figures/taylor.svg)

First order justifies gradient descent (the loss decreases along $-\nabla f$ for a small enough step); second order justifies Newton's method and explains why curvature limits the learning rate. Useful expansions: $e^x = \sum x^n/n!$; $\ln(1+x) \approx x - x^2/2$; $\sqrt{1+x} \approx 1 + x/2$.

## Integrals

$$\int_a^b f(x)\,dx = F(b) - F(a), \qquad F' = f \quad \text{(fundamental theorem of calculus)}$$

| $f(x)$ | $\int f\,dx$ | Technique | Idea |
|---|---|---|---|
| $x^n,\ n\ne-1$ | $\frac{x^{n+1}}{n+1}+C$ | substitution | reverse chain rule: $\int f(g(x))g'(x)dx = \int f(u)du$ |
| $1/x$ | $\ln\lvert x\rvert + C$ | by parts | $\int u\,dv = uv - \int v\,du$ |
| $e^{ax}$ | $e^{ax}/a + C$ | partial fractions | split rational functions |
| $\int_{-\infty}^{\infty}e^{-x^2}dx$ | $\sqrt\pi$ | polar coordinates | normalising the Gaussian |
| $\int_0^\infty \lambda e^{-\lambda x}dx$ | 1 | — | the exponential pdf integrates to 1 |

!!! example "Worked example: expectation by parts"
    $\mathbb{E}[X]$ for $X \sim \text{Exp}(\lambda)$: $\int_0^\infty x\lambda e^{-\lambda x}dx$. With $u = x$, $dv = \lambda e^{-\lambda x}dx$: $= \big[-xe^{-\lambda x}\big]_0^\infty + \int_0^\infty e^{-\lambda x}dx = 0 + 1/\lambda$.

**In ML:** probabilities are areas under densities; expectations $\mathbb{E}[g(X)] = \int g(x)p(x)dx$; normalising constants (Bayesian evidence); AUC is an integral; ELBO in VAEs. When integrals are intractable: Monte Carlo ($\frac1N\sum g(x_i)$, error $\propto 1/\sqrt N$), quadrature, or variational approximations.

```python
from scipy import integrate; integrate.quad(lambda x: x**2, 0, 2)      # (2.6667, 3e-14)
import sympy as sp; x = sp.symbols("x")
sp.diff(sp.sin(x) * x**2, x); sp.integrate(x * sp.exp(-x), (x, 0, sp.oo)); sp.limit(sp.sin(x) / x, x, 0)
```

## Multivariable calculus: gradient, Jacobian, Hessian

$$\nabla f = \begin{pmatrix}\frac{\partial f}{\partial x_1}\\ \vdots\\ \frac{\partial f}{\partial x_d}\end{pmatrix},\qquad J_{ij} = \frac{\partial f_i}{\partial x_j},\qquad H_{ij} = \frac{\partial^2 f}{\partial x_i\,\partial x_j}$$

| Object | Shape | Meaning | ML use |
|---|---|---|---|
| Gradient $\nabla f$ | $d$ | direction of steepest **ascent**; $\lVert\nabla f\rVert$ = steepness | every optimiser |
| Directional derivative | scalar $\nabla f^\top\mathbf{u}$ | slope along unit vector $\mathbf{u}$ | line search |
| Jacobian $J$ | $m\times d$ | best linear approximation of $f:\mathbb{R}^d\to\mathbb{R}^m$ | backprop chains Jacobian–vector products; normalising flows ($\log\lvert\det J\rvert$) |
| Hessian $H$ | $d\times d$ | curvature; symmetric for smooth $f$ | Newton's method, saddle detection, Laplace approximation |
| Second-order test | eigenvalues of $H$ at $\nabla f = 0$ | all $>0$ min, all $<0$ max, mixed saddle | non-convex landscapes |

**Matrix-calculus identities** (denominator layout): $\nabla_\mathbf{x}(\mathbf{a}^\top\mathbf{x}) = \mathbf{a}$; $\nabla_\mathbf{x}(\mathbf{x}^\top A\mathbf{x}) = (A + A^\top)\mathbf{x}$; $\nabla_\mathbf{w}\lVert X\mathbf{w}-\mathbf{y}\rVert^2 = 2X^\top(X\mathbf{w}-\mathbf{y})$; $\nabla_X\operatorname{tr}(AX) = A^\top$; $\nabla_X\log\det X = X^{-\top}$.

!!! example "Worked example: gradient and Hessian"
    $f(x,y) = x^2 + 3xy + 2y^2$. $\nabla f = (2x + 3y,\ 3x + 4y)$. At $(1,1)$: $\nabla f = (5, 7)$. $H = \begin{pmatrix}2&3\\3&4\end{pmatrix}$, $\det H = -1 < 0$, so the stationary point at the origin is a **saddle**.

## Key derivations for ML

!!! derivation "Derivation: gradient of the logistic loss"
    For one example, $p = \sigma(z)$, $z = \mathbf{w}^\top\mathbf{x}$, $\ell = -[y\ln p + (1-y)\ln(1-p)]$. Then $\frac{\partial\ell}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p}$ and $\frac{\partial p}{\partial z} = p(1-p)$, so $\frac{\partial\ell}{\partial z} = -y(1-p) + (1-y)p = p - y$. With $\frac{\partial z}{\partial\mathbf{w}} = \mathbf{x}$: $\nabla_\mathbf{w}\ell = (p - y)\,\mathbf{x}$. Over the dataset: $\nabla\mathcal{L} = \frac1n X^\top(\sigma(X\mathbf{w}) - \mathbf{y})$. The Hessian $\frac1n X^\top SX$ with $S = \operatorname{diag}(p_i(1-p_i))$ is PSD, so the loss is **convex**.

@@code code/s04_gradcheck.py@@
@@out code/out/s04_gradcheck.txt@@

!!! derivation "Derivation: softmax + cross-entropy gradient"
    $p_k = e^{z_k}/\sum_j e^{z_j}$, $\ell = -\sum_k y_k\ln p_k$ with one-hot $\mathbf{y}$. The softmax Jacobian is $\frac{\partial p_k}{\partial z_j} = p_k(\delta_{kj} - p_j)$. Hence $\frac{\partial\ell}{\partial z_j} = -\sum_k \frac{y_k}{p_k}p_k(\delta_{kj} - p_j) = -y_j + p_j\sum_k y_k = p_j - y_j$. The gradient is simply **prediction minus target**: $\nabla_\mathbf{z}\ell = \mathbf{p} - \mathbf{y}$. That's why frameworks fuse softmax and cross-entropy (`CrossEntropyLoss` takes logits): it is cheaper and numerically stable (log-sum-exp).

!!! derivation "Derivation: bias–variance decomposition"
    Let $y = f(\mathbf{x}) + \varepsilon$ with $\mathbb{E}[\varepsilon] = 0$, $\operatorname{Var}(\varepsilon) = \sigma^2$, and let $\hat f$ be trained on a random dataset $D$. With $\bar f = \mathbb{E}_D[\hat f]$: $\mathbb{E}\big[(y - \hat f)^2\big] = \mathbb{E}\big[(f + \varepsilon - \bar f + \bar f - \hat f)^2\big] = \underbrace{(f - \bar f)^2}_{\text{bias}^2} + \underbrace{\mathbb{E}_D[(\hat f - \bar f)^2]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{noise}}$. The cross terms vanish because $\varepsilon$ is independent of $D$ with mean zero, and $\mathbb{E}_D[\hat f - \bar f] = 0$.

## Gradient descent and its variants

$$\theta_{t+1} = \theta_t - \eta\,\nabla_\theta\mathcal{L}(\theta_t)$$

![](figures/gradient_descent.png)

!!! derivation "Why the learning rate is bounded by curvature"
    On a quadratic $\mathcal{L} = \frac12\lambda\theta^2$, one step gives $\theta_{t+1} = (1 - \eta\lambda)\theta_t$. It converges iff $\lvert1 - \eta\lambda\rvert < 1 \iff \eta < 2/\lambda$. In many dimensions the largest Hessian eigenvalue $\lambda_{\max}$ sets the limit $\eta < 2/\lambda_{\max}$, while the smallest sets the speed, so convergence takes about $\kappa = \lambda_{\max}/\lambda_{\min}$ steps. Feature scaling, normalisation layers and adaptive optimisers all attack $\kappa$.

![](figures/optimizers.svg)

| Optimiser | Update rule | Defaults | When |
|---|---|---|---|
| SGD | $\theta \leftarrow \theta - \eta g$ | $\eta$ by LR finder | convex problems; strong generalisation for CNNs with schedules |
| Momentum | $v \leftarrow \beta v + g$; $\theta \leftarrow \theta - \eta v$ | $\beta = 0.9$ | ravines; standard for ResNets |
| Nesterov | gradient at the look-ahead point $\theta - \eta\beta v$ | $\beta = 0.9$ | slightly faster momentum |
| AdaGrad | $\theta \leftarrow \theta - \eta g/\sqrt{\sum g^2}$ | $\eta = 0.01$ | sparse features; LR decays to 0 |
| RMSProp | $s \leftarrow \rho s + (1-\rho)g^2$; $\theta \leftarrow \theta - \eta g/\sqrt{s+\epsilon}$ | $\rho = 0.9$ | RNNs, non-stationary objectives |
| **Adam** | $m \leftarrow \beta_1 m + (1-\beta_1)g$; $v \leftarrow \beta_2 v + (1-\beta_2)g^2$; $\hat m = \frac{m}{1-\beta_1^t}$, $\hat v = \frac{v}{1-\beta_2^t}$; $\theta \leftarrow \theta - \eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}$ | $\eta = 10^{-3}$, $\beta = (0.9, 0.999)$, $\epsilon = 10^{-8}$ | default for most deep learning |
| **AdamW** | Adam + decoupled decay $\theta \leftarrow \theta - \eta\lambda\theta$ | $\lambda = 0.01$–$0.1$ | transformers; default in modern training |
| Newton | $\theta \leftarrow \theta - H^{-1}g$ | — | small, smooth, convex problems; quadratic convergence |
| L-BFGS | low-rank $H^{-1}$ approximation | memory 10–20 | full-batch smooth problems (sklearn `LogisticRegression`) |

![](figures/lr_effects.svg)

![](figures/lr_schedules.svg)

!!! rule "Rule of thumb: learning rate"
    AdamW: start at $3\times10^{-4}$ (transformers $1$–$5\times10^{-4}$ with warm-up; fine-tuning $1$–$5\times10^{-5}$). SGD with momentum: $0.1$ for ResNets with batch 256, scaled linearly with batch size. Run an LR range test and choose about 10× below the point where the loss explodes.

## Convexity, constraints and the loss landscape

![](figures/landscapes.png)

- **Convex:** $f(\lambda\mathbf{a} + (1-\lambda)\mathbf{b}) \le \lambda f(\mathbf{a}) + (1-\lambda)f(\mathbf{b})$; equivalently $H \succeq 0$. Every local minimum is global (linear/logistic regression, SVM, lasso).
- **Non-convex** (neural networks): many minima, mostly of similar quality in high dimensions; **saddle points** and flat plateaus are the real obstacles. Noise from SGD and momentum help escape them. Flat minima tend to generalise better than sharp ones.
- **Lagrange multipliers:** to optimise $f$ subject to $g(\mathbf{x}) = 0$, solve $\nabla f = \lambda\nabla g$. **KKT** conditions extend this to inequalities: stationarity, primal/dual feasibility, complementary slackness $\mu_i g_i(\mathbf{x}) = 0$ (the SVM support vectors are exactly the points with $\mu_i > 0$).
- **Projected / proximal methods** handle constraints and non-smooth penalties: soft-thresholding is the proximal operator of L1, $\operatorname{prox}(w) = \operatorname{sign}(w)\max(\lvert w\rvert - \eta\lambda, 0)$.

!!! interview "Interview question: why does Adam sometimes generalise worse than SGD?"
    Adam's per-parameter scaling lets it converge fast to sharper minima, and coupling L2 regularisation with adaptive scaling weakens the regularisation (fixed by AdamW). For vision models trained from scratch, well-tuned SGD + momentum + cosine schedule often generalises slightly better; for transformers, AdamW with warm-up is standard. In practice the schedule and weight decay matter more than the optimiser choice.

<div class="card" markdown="1">

## Cheat card: Calculus & optimisation

<div class="cols" markdown="1">

#### Derivatives to know cold
- $(x^n)' = nx^{n-1}$, $(e^x)' = e^x$, $(\ln x)' = 1/x$
- $\sigma' = \sigma(1-\sigma)$, $\tanh' = 1-\tanh^2$
- chain: $(f\circ g)' = f'(g)\,g'$

#### Gradients of classic losses
- MSE: $\frac2n X^\top(X\mathbf{w}-\mathbf{y})$
- logistic: $\frac1n X^\top(\sigma(X\mathbf{w})-\mathbf{y})$
- softmax-CE w.r.t. logits: $\mathbf{p}-\mathbf{y}$

#### Curvature
- Hessian PD → minimum; mixed → saddle
- stable step: $\eta < 2/\lambda_{\max}$
- condition number $\kappa$ = steps needed

#### Optimiser defaults
- AdamW, lr 3e-4, wd 0.01, warm-up + cosine
- SGD + momentum 0.9 for CNNs
- clip grad-norm at 1.0

#### Debug with a gradient check
- central difference, $\epsilon = 10^{-6}$
- relative error $< 10^{-6}$ ✓, $> 10^{-3}$ ✗
- use float64 for the check

#### Integrals
- probability = area under pdf
- $\mathbb{E}[g(X)] = \int g(x)p(x)dx$
- intractable → Monte Carlo ($1/\sqrt N$)

</div>
</div>
