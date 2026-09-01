#!/usr/bin/env python3
"""Generate a fixed sequence of K perturbed expected-return vectors (mu_t) so both
solvers re-optimize the IDENTICAL sequence. mu_t = mu * (1 + 2% i.i.d.)."""
import numpy as np
D = np.load("/scratch/headtohead/portfolio.npz"); mu = D["mu"]; n = int(D["n"])
K = 20
rng = np.random.default_rng(12345)
mu_seq = mu[None, :] * (1.0 + 0.02 * (2.0 * rng.random((K, n)) - 1.0))
np.savez("/scratch/headtohead/qpbench/resolve_seq.npz", mu_seq=mu_seq.astype(np.float64), K=K)
print(f"wrote resolve_seq.npz  K={K}  n={n}  mu_seq{mu_seq.shape}")
