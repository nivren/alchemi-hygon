# BFGS & Quasi-Newton Spectral Optimization on DCU

## Background: Why BFGS Needs Batched Eigh

In crystal structure relaxation, molecular geometry optimization, and potential
energy surface exploration, Quasi-Newton algorithms update an estimate of the
Hessian matrix $H_k \in \mathbb{R}^{N \times N}$ or its inverse
$B_k = H_k^{-1}$, where $N = 3 \times \text{num\_atoms}$.

When optimizing a batch of structures concurrently:
- Typical batch size $B \in [4, 32]$.
- Degrees of freedom $N \in [100, 1000]$.

The algorithm regularly performs:
$$H_k = V \Lambda V^T = \sum_{i=1}^N \lambda_i v_i v_i^T$$

### Quasi-Newton Step Formulation

1. Regularize curvature:
   $$p_k = - V \text{diag}\left(\frac{1}{\max(\lambda_i, \epsilon_{\min})}\right) V^T g_k$$
2. Use a batched eigensolver to obtain $w$ and $V$.
3. Project the gradient with $V^T g_k$.
4. Clamp eigenvalues and rotate the step back with $V$.

The source implementation pattern is:

    w, V = eigh_batched(H_batch, UPLO="L")
    alpha = torch.bmm(V.transpose(-1, -2), g_batch)
    w_reg = torch.clamp(w, min=min_eig).unsqueeze(-1)
    step = -torch.bmm(V, alpha / w_reg)

### Performance Evidence Boundary

The source project reports 15x to 27x eigensolver speedups for its own
torch_rocsolver_eigh package and workload. Treat those numbers as source
benchmark evidence; remeasure them for this repository, framework path, target
device, dtype, and batch shape before using them in a support claim.
