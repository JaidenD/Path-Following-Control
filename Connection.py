from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from Manifolds.manifold import RiemannianManifold


def lc_christoffel(
    Q: "RiemannianManifold",
    q: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Computes Levi-Civita Christoffel symbols at q

    Parameters
    ----------
    Q: RiemannianManifold
    q: Coordinate point where the christoffel symbol is evaluated
    eps: Finite difference step size
    """
    n = Q.dim
    q = np.asarray(q, dtype=float)

    g = Q.metric(q)
    g_inv = np.linalg.inv(g)

    dg = np.zeros((n, n, n))

    for a in range(n):
        dq = np.zeros(n)
        dq[a] = eps

        g_plus = Q.metric(q + dq)
        g_minus = Q.metric(q - dq)

        dg[a] = (g_plus - g_minus) / (2.0 * eps)
    Gamma = np.zeros((n, n, n))

    for k in range(n):
        for i in range(n):
            for j in range(n):
                total = 0.0
                for ell in range(n):
                    total += g_inv[k, ell] * (
                        dg[i, j, ell] + dg[j, i, ell] - dg[ell, i, j]
                    )
                Gamma[k, i, j] = 0.5 * total
    return Gamma
    
