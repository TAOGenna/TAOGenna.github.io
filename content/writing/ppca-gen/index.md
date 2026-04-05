---
title: "PPCA: The Minimal Generative Model"
date: 2025-10-20
categories: ["AI"]
---

# PPCA: Probabilistic Principal Component Analysis  

PCA is one of those algorithms you learn early and then treat as a black box: center the data, compute the top eigenvectors of the sample covariance, project. But PCA *feels* algebraic, where does it come from probabilistically? Probabilistic PCA (PPCA) answers that question: it is the **maximum-likelihood solution of a simple linear Gaussian latent-variable model**. Understanding PPCA is a tiny, high-value step on the path to modern generative models (VAEs, normalizing flows): it isolates the linear + Gaussian case where every calculation is analytic, so you can see clearly how inference and likelihood tie together.

Below I present a single self-contained story: model → marginal → posterior → likelihood → MLE solution → interpretation. I'll highlight the key identities you need and keep the algebra explicit so nothing mysterious is swept under the rug.

---

# Motivation

We want a generative model for $d$-dimensional observations $t$ that:

- Is simple enough to solve analytically; and
- Reduces to classical PCA in an appropriate limit.

A latent-variable model does exactly this: introduce a low-dimensional latent $x\in\mathbb{R}^q$ ($q<d$) and pick simple Gaussian forms so all integrals and conditionals are Gaussian.

---

# Model (linear Gaussian latent variable)

Assume the generative process
$$
x \sim \mathcal{N}(0, I_q),\qquad
t \mid x \sim \mathcal{N}(W x + \mu,\, \sigma^2 I_d),
$$
where $W$ is a $d\times q$ matrix (the linear mapping), $\mu\in\mathbb{R}^d$ is the mean, and $\sigma^2>0$ is isotropic noise variance. This is the PPCA model (Tipping & Bishop, 1999).

Two immediate facts:

1. Because both prior and likelihood are Gaussian, the marginal $p(t)$ and posterior $p(x\mid t)$ are Gaussian and computable in closed form.  
2. The model has a rotational ambiguity in $W$: replacing $W$ by $W R$ for any orthogonal $R\in\mathbb{R}^{q\times q}$ leaves the model unchanged (with appropriate change of latent coordinates). So $W$ is identifiable only up to orthogonal transforms.

For the sake of clarity I'll use the explanation of [Oliver](https://medium.com/practical-coding/the-simplest-generative-model-you-probably-missed-c840d68b704):

{{< figure src="ppca_prince.jpg" alt="PPCA diagram" caption="Bishop's \"Pattern Recognition and Machine Learning\", chapter 12." class="align-center" >}}

- Left panel: After sampling a variable from the latent distribution,
- Middle panel: The visibles are drawn from an isotropic Gaussian (diagonal covariance matrix) around $W * x_h + \mu$.
- Right panel: The resulting marginal distribution for the observables is also a Gaussian, but not isotropic.

---

# Marginal distribution $p(t)$

We compute the marginal by integrating out $x$:
$$
p(t) = \int p(t\mid x)\,p(x)\,dx.
$$
Set $r := t-\mu$. Using the Gaussian forms,
$$
p(t\mid x)\,p(x)
\propto
\exp\Big(-\tfrac{1}{2}\big[(r - W x)^\top (\sigma^{-2} I_d)(r - W x) + x^\top x\big]\Big).
$$
Collect the quadratic terms in $x$:

$$
\frac{1}{\sigma^{2}}(r - W x)^{T}(r - W x) + x^{T}x = \frac{1}{\sigma^{2}} r^{T}r - \frac{2}{\sigma^{2}} x^{T} W^{T} r + x^{T}\Big(I_q + \frac{1}{\sigma^{2}} W^{T} W\Big)x.
$$

Define the $q\times q$ matrix
$$
M := I_q + \frac{1}{\sigma^2} W^{T}W.
$$
Completing the square in $x$ gives a Gaussian integral whose value is $(2\pi)^{q/2} |M|^{-1/2}$. Carefully tracking normalization constants (from $p(x)$ and $p(t\mid x)$) yields
$$
p(t)
= \frac{1}{(2\pi)^{d/2} (\sigma^2)^{d/2} } |M|^{-1/2}
\exp\Big(-\tfrac{1}{2} r^\top \big(\tfrac{1}{\sigma^2} I_d - \tfrac{1}{\sigma^4} W M^{-1} W^\top\big) r\Big).
$$

Now recognize the familiar covariance and precision using two matrix identities.

## Woodbury identity (precision)
For $C := W W^\top + \sigma^2 I_d$,
$$
C^{-1} = (\sigma^2 I_d + W W^\top)^{-1}
= \frac{1}{\sigma^2} I_d - \frac{1}{\sigma^4} W\big(I_q + \tfrac{1}{\sigma^2}W^\top W\big)^{-1}W^\top
= \frac{1}{\sigma^2} I_d - \frac{1}{\sigma^4} W M^{-1} W^\top.
$$
This matches the precision we found in the exponent.

## Matrix determinant lemma
$$
|W W^\top + \sigma^2 I_d| = (\sigma^2)^d\,\Big|I_q + \frac{1}{\sigma^2}W^\top W\Big| = (\sigma^2)^d |M|.
$$
Therefore $(\sigma^2)^{d/2}|M|^{1/2}=|C|^{1/2}$.

Putting the pieces together,
$$
\boxed{p(t) = \mathcal{N}(t\mid \mu,\, C), \qquad
C := W W^\top + \sigma^2 I_d. }
$$

So PPCA is a Gaussian density with covariance equal to the low-rank part $W W^\top$ plus isotropic noise.

---

# Posterior $p(x\mid t)$ (exact inference)

From the quadratic completion above one also reads the posterior over $x$:
$$
p(x\mid t) = \mathcal{N}\big(x \mid m, \Sigma\big)
$$
with
$$
\boxed{ \Sigma = \sigma^2 \big(W^\top W + \sigma^2 I_q\big)^{-1}, \qquad
m = \big(W^\top W + \sigma^2 I_q\big)^{-1} W^\top (t-\mu).}
$$
(Equivalently, define $M := I_q + \sigma^{-2}W^\top W$ and you get $\Sigma=M^{-1}$ and $m = M^{-1}\sigma^{-2}W^\top (t-\mu)$; both forms are algebraically the same.)

Two intuitions:

- When $\sigma^2$ is small, posterior variance is small and $m\approx (W^\top W)^{-1}W^\top (t-\mu)$, i.e. classical least-squares projection onto the column space of $W$.  
- When $\sigma^2$ is large, the posterior concentrates near zero (the prior) because the observation is noisy.

---

# Likelihood for a dataset and maximum likelihood estimation

Given $N$ i.i.d. datapoints $\lbrace t_i\rbrace_{i=1}^N$, 

$$
\begin{align*}
p(T | W,\psi, \mu) &= \prod_{i=1}^N p(t_i | W,\psi, \mu)\\
&= (2\pi)^{-ND/2}|C|^{-N/2}\exp{\Big\lbrace -\frac{1}{2}\sum_{i=1}^N (t_i-\mu)^TC^{-1}(t_i-\mu) \Big\rbrace}
\end{align*}
$$

the log-likelihood is
$$
\log p(T\mid W,\sigma^2,\mu)
= -\frac{N}{2}\Big( d\log(2\pi) + \log|C| + \operatorname{Tr}\big(C^{-1} S\big) \Big),
$$
where $S$ is the sample covariance

$$
S := \frac{1}{N}\sum_{i=1}^N (t_i-\mu)(t_i-\mu)^\top.
$$

For fixed $\mu$ (choose $\mu=\bar t$), maximizing the likelihood reduces to minimizing
$$
L(W,\sigma^2) =\log|C| + \operatorname{Tr}(C^{-1} S), \qquad C = W W^\top + \sigma^2 I_d.
$$

Tipping & Bishop show that the stationary point can be written in closed form using the top eigenstructure of $S$. Let the eigen-decomposition be
$$
S = U \Lambda U^\top,\qquad \Lambda = \operatorname{diag}(\lambda_1,\dots,\lambda_d),\quad \lambda_1\ge\cdots\ge\lambda_d.
$$
Split $U = [U_q\ U_{d-q}]$ and $\Lambda = \operatorname{diag}(\Lambda_q,\Lambda_{d-q})$ where $U_q$ are the top $q$ eigenvectors and $\Lambda_q$ the corresponding eigenvalues. Then the maximum-likelihood estimates are

$$
\boxed{
\begin{aligned}
\mu^* &= \bar t,\\[4pt]
W^* &= U_q\big(\Lambda_q - \sigma^{2*} I_q\big)^{1/2} R,\qquad R\in\mathbb{R}^{q\times q}\ \text{orthogonal},\\[4pt]
\sigma^{2*} &= \frac{1}{d-q}\sum_{j=q+1}^{d}\lambda_j.
\end{aligned}
}
$$


A few remarks:

- The orthogonal matrix $R$ remains free: it is the rotational indeterminacy of the latent coordinates.  
- The noise variance $\sigma^{2*}$ is the average of the discarded eigenvalues (the variance not explained by the principal subspace).  
- When $\sigma^2\to 0$, the columns of $W^*$ span the same subspace as the top-$q$ PCA directions and $W^*W^{*\top}$ tends to $U_q\Lambda_q U_q^\top$. Thus PPCA recovers classical PCA as a limiting case.

If you want to compute the MLE in practice:

- You can directly solve for $\sigma^{2*}$ from the eigenvalues and then compute $W^*$ from the top eigenpairs (this is Tipping & Bishop's closed form).  
- Alternatively, you can optimize $L(W,\sigma^2)$ numerically or use an EM algorithm (below) if you prefer an iterative approach that cleanly separates the inference (E-step) and parameter update (M-step).

---

# EM algorithm

EM treats $x$ as missing data. With current parameters $(W,\sigma^2)$ compute posterior expectations
$$
\mathbb{E}[x\mid t] = m,\qquad \mathbb{E}[x x^\top \mid t] = \Sigma + m m^\top.
$$
Then M-step updates $W$ and $\sigma^2$ by solving linear equations involving these expectations (closed form). EM is useful when you want a monotonic increase in likelihood or when you want to generalize (e.g. to non-isotropic noise $\Psi$) where closed forms are less tidy.

---

# PPCA vs Factor Analysis vs PCA

- **Classical PCA**: algorithmic; finds subspace spanned by top eigenvectors of $S$. No generative likelihood (no noise model).  
- **PPCA**: linear generative model with *isotropic* Gaussian noise $\sigma^2 I_d$. Likelihood is available and PCA arises in the limit $\sigma^2\to 0$.  
- **Factor analysis (FA)**: same linear structure but with *diagonal* noise covariance $\Psi$ (not necessarily isotropic). FA is more flexible but algebraically messier, closed-form MLEs for $\Psi$ do not generally exist and EM is standard.

---

# Sampling & use as a generative model

Sampling from PPCA is trivial and instructive:

1. Sample $x\sim\mathcal{N}(0,I_q)$.
2. Sample noise $\epsilon\sim\mathcal{N}(0,\sigma^2 I_d)$.
3. Return $t = W x + \mu + \epsilon$.

This generates samples from $\mathcal{N}(\mu, W W^\top + \sigma^2 I_d)$. So PPCA is *not* a complex high-capacity generative model, but it provides a clear probabilistic story for projection + reconstruction and clarifies the role of latent dimensions and noise.

# Code and visuals

{{< figure src="ppca_output.png" alt="PPCA output" caption="We start from the observed data (in blue) and through the PPCA max log likelihood we find the optimized parameters, which once plugged in the linear relation and sample some values from the latent space we effectively generate a new dataset (in red) with similar distribution as the observed dataset." class="align-center" >}}

```python
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh

# -------------------------------------
# Step 1. Generate some real 2D data
# -------------------------------------
np.random.seed(0)
N = 500
z = np.random.randn(N, 1)
T_real = np.hstack([z, 0.5*z + 0.2*np.random.randn(N, 1)])  # correlated data

# -------------------------------------
# Step 2. Estimate PPCA parameters
# -------------------------------------
# Center data
mu = T_real.mean(axis=0)
X_centered = T_real - mu

# Sample covariance
S = np.cov(X_centered.T)

# Eigen decomposition
vals, vecs = eigh(S)
idx = np.argsort(vals)[::-1]
vals, vecs = vals[idx], vecs[:, idx]

# Latent dimension q = 1
q = 1
U_q = vecs[:, :q]
Lambda_q = vals[:q]

# Compute sigma^2 and W
D = T_real.shape[1]
sigma2 = np.mean(vals[q:])  # average of remaining eigenvalues
W = U_q * np.sqrt(Lambda_q - sigma2)

print(f"Learned W =\n{W}")
print(f"Learned sigma^2 = {sigma2:.4f}")
print(f"Mean = {mu}")

# -------------------------------------
# Step 3. Generate new data from model
# -------------------------------------
N_gen = 500
x_gen = np.random.randn(N_gen, q)
eps = np.sqrt(sigma2) * np.random.randn(N_gen, D)
T_gen = x_gen @ W.T + mu + eps

# -------------------------------------
# Step 4. Compare
# -------------------------------------
plt.figure(figsize=(6,6))
plt.scatter(T_real[:,0], T_real[:,1], alpha=0.5, label="Real data", color="royalblue")
plt.scatter(T_gen[:,0], T_gen[:,1], alpha=0.5, label="Generated from PPCA", color="tomato")
plt.title("PPCA as a Generative Model")
plt.legend()
plt.axis("equal")
plt.show()
```


---

# Intuition & how to read the equations

- $W W^\top$ captures the **structured, low-rank variability** in the data (the principal subspace).  
- $\sigma^2 I_d$ captures **isotropic residual noise**, variance not explained by the subspace.  
- The posterior $p(x\mid t)$ tells you how to infer latent coordinates of a datum: as noise decreases ($\sigma^2\to0$) the posterior collapses to a point and the mean becomes the usual linear projection onto principal components.  
- The MLE for $\sigma^2$ being the mean of the discarded eigenvalues gives a neat decomposition: total variance = explained (top $q$ eigenvalues) + unexplained (average of remaining eigenvalues).

---

# Bridge to VAEs (why study PPCA first)

A Variational Autoencoder (VAE) generalizes PPCA in two directions:

- Replace the linear decoder $W x + \mu$ by a neural network $f_\theta(x)$ (nonlinear decoder).  
- Replace exact Gaussian posterior inference (available in PPCA) with an *approximate* amortized posterior $q_\phi(x\mid t)$ (the encoder), trained by variational inference.

PPCA is therefore the canonical linear Gaussian toy model where inference, sampling, and MLE are closed form. Studying PPCA first gives you a clean map of what changes when you make the model nonlinear and inference approximate.

# References
[1] M. E. Tipping and C. M. Bishop, "Probabilistic Principal Component Analysis," Journal of the Royal Statistical Society: Series B (Statistical Methodology) 61, 611-622 (1999), https://www.cs.columbia.edu/~blei/seminar/2020-representation/readings/TippingBishop1999.pdf  

[2] S. Patel, "The Simplest Generative Model You Probably Missed," Medium (2018), https://medium.com/practical-coding/the-simplest-generative-model-you-probably-missed-c840d68b704  

[3] S. J. D. Prince, *Understanding Deep Learning* (MIT Press, 2023), https://udlbook.github.io/udlbook/
