from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp

from Connection import lc_christoffel

if TYPE_CHECKING:
    from Manifolds.manifold import RiemannianManifold


def geodesic_rhs(Q: "RiemannianManifold", eps: float = 1e-6):
    """
    Return vectorized RHS for the geodesic equation.

    State y = [q, v] expressed in local chart 
    """

    n = Q.dim

    def rhs(t: np.ndarray, y: np.ndarray) -> np.ndarray:
        m = y.shape[1]

        dy = np.zeros_like(y)

        for a in range(m):
            q = y[:n, a]
            v = y[n:, a]

            Gamma = lc_christoffel(Q, q, eps=eps)

            acc = np.zeros(n)

            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        acc[k] -= Gamma[k, i, j] * v[i] * v[j]
            dy[:n, a] = v
            dy[n:, a] = acc
        return dy
    
    return rhs


def geodesic_bc(q0: np.ndarray, q1: np.ndarray):
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    n = q0.shape[0]

    def bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        return np.concatenate([
            ya[:n] - q0,
            yb[:n] - q1
        ])
    
    return bc


def solve_geodesic_bvp(
    Q: "RiemannianManifold",
    q0: np.ndarray,
    q1: np.ndarray,
    eps: float = 1e-6,
    num_nodes: int = 20,
    tol: float = 1e-5,
    max_nodes: int = 1000,
):
    """
    Solve the coordinate geodesic BVP from q0 to q1.

    Returns
    -------
    sol:
        scipy BVP solution
    xi: np.ndarray
        Initial velocity v(0), representing Log_q0(q1)
    """
    n = Q.dim
    q0 = Q.coordinate(q0)
    q1 = Q.coordinate(q1)

    # Work in an unwrapped local chart so revolute joints can choose the short
    # displacement branch while the metric still sees projected coordinates.
    dq = Q.Q.log(q0, q1)
    q1_chart = q0 + dq
    t = np.linspace(0.0, 1.0, num_nodes)

    q_guess = np.zeros((n, num_nodes))
    v_guess = np.zeros((n, num_nodes))

    for a in range(num_nodes):
        q_guess[:, a] = (1.0 - t[a]) * q0 + t[a] * q1_chart
        v_guess[:, a] = dq
    y_guess = np.vstack([q_guess, v_guess])

    rhs = geodesic_rhs(Q, eps=eps)
    bc = geodesic_bc(q0, q1_chart)

    sol = solve_bvp(
        rhs,
        bc,
        t,
        y_guess,
        tol=tol,
        max_nodes=max_nodes,
    )

    if not sol.success:
        raise RuntimeError(f"Geodesic BVP failed: {sol.message}")
    
    n = Q.dim
    xi = sol.y[n:, 0]

    return sol, xi


def solve_geodesic_ivp(
    Q: "RiemannianManifold",
    q0: np.ndarray,
    v0: np.ndarray,
    eps: float = 1e-6,
    tol: float = 1e-7,
    max_step: float = 0.05,
) -> np.ndarray:
    """Integrate the geodesic IVP for one unit of affine time."""
    n = Q.dim
    q0 = Q.coordinate(q0)
    v0 = np.asarray(v0, dtype=float).reshape(n)
    y0 = np.concatenate([q0, v0])
    rhs = geodesic_rhs(Q, eps=eps)

    sol = solve_ivp(
        lambda t, y: rhs(np.array([t]), y.reshape(2 * n, 1)).reshape(2 * n),
        (0.0, 1.0),
        y0,
        rtol=tol,
        atol=tol,
        max_step=max_step,
    )

    if not sol.success:
        raise RuntimeError(f"Geodesic IVP failed: {sol.message}")

    return Q.project(sol.y[:n, -1])
