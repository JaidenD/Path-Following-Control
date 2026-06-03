from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

import numpy as np


MetricLike = Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]


class Manifold(ABC):
    """
    Minimal coordinate-manifold interface used by the controllers.
    """

    dim: int

    @abstractmethod
    def coordinate(self, q: np.ndarray) -> np.ndarray:
        """Return a coordinate representation of ``q``."""

    def project(self, q: np.ndarray) -> np.ndarray:
        """Project coordinates onto the represented manifold."""
        return self.coordinate(q)

    def exp(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Default chart exponential used for flat manifolds."""
        return self.project(np.asarray(q, dtype=float) + np.asarray(v, dtype=float))

    def log(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Default chart logarithm used for flat manifolds."""
        return self.coordinate(p) - self.coordinate(q)

    def interpolate(self, q: np.ndarray, p: np.ndarray, u: float) -> np.ndarray:
        return self.exp(q, float(u) * self.log(q, p))

    def dist(self, q: np.ndarray, p: np.ndarray) -> float:
        return float(np.linalg.norm(self.log(q, p)))


class Euclidean(Manifold):
    """Euclidean configuration space R^n."""

    def __init__(self, dim: int):
        if dim <= 0:
            raise ValueError("Euclidean dimension must be positive.")
        self.dim = int(dim)

    def coordinate(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape != (self.dim,):
            raise ValueError(f"Expected point with shape ({self.dim},), got {q.shape}.")
        return q


class RiemannianManifold:
    """A coordinate manifold equipped with a Riemannian metric.

    If the metric is supplied as a constant matrix, geodesics are taken from the
    underlying manifold's analytic chart operations. If the metric is callable,
    ``Log`` and ``Exp`` use numerical geodesic solvers by default. Pass
    ``use_analytic=True`` to force the underlying manifold's flat operations.
    """

    def __init__(
        self,
        manifold: Manifold,
        metric: Optional[MetricLike] = None,
        *,
        use_analytic: Optional[bool] = None,
    ):
        self.Q = manifold
        self.dim = manifold.dim

        if metric is None:
            self._constant_metric = np.eye(self.dim)
            self._metric_fn = None
        elif callable(metric):
            self._constant_metric = None
            self._metric_fn = metric
        else:
            self._constant_metric = np.asarray(metric, dtype=float)
            self._metric_fn = None

        test_metric = self.metric(np.zeros(self.dim))
        if test_metric.shape != (self.dim, self.dim):
            raise ValueError(
                f"Metric must have shape ({self.dim}, {self.dim}), got {test_metric.shape}."
            )

        if not np.allclose(test_metric, test_metric.T, atol=1e-10):
            raise ValueError("Metric must be symmetric.")

        self.use_analytic = self._constant_metric is not None if use_analytic is None else bool(use_analytic)
        self._solver_diagnostics = {
            "bvp_calls": 0,
            "ivp_calls": 0,
        }

    def reset_solver_diagnostics(self) -> None:
        self._solver_diagnostics["bvp_calls"] = 0
        self._solver_diagnostics["ivp_calls"] = 0

    def solver_diagnostics(self) -> dict[str, int]:
        return dict(self._solver_diagnostics)

    def coordinate(self, q: np.ndarray) -> np.ndarray:
        return self.Q.coordinate(q)

    def project(self, q: np.ndarray) -> np.ndarray:
        return self.Q.project(q)

    def metric(self, q: np.ndarray) -> np.ndarray:
        if self._constant_metric is not None:
            return self._constant_metric
        q_projected = self.project(q)
        g = np.asarray(self._metric_fn(q_projected), dtype=float)
        if g.shape != (self.dim, self.dim):
            raise ValueError(f"Metric returned shape {g.shape}; expected ({self.dim}, {self.dim}).")
        return g

    def inner(self, q: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
        v = np.asarray(v, dtype=float).reshape(self.dim)
        w = np.asarray(w, dtype=float).reshape(self.dim)
        return float(v @ self.metric(q) @ w)

    def norm(self, q: np.ndarray, v: np.ndarray) -> float:
        val = self.inner(q, v, v)
        return float(np.sqrt(max(val, 0.0)))

    def squared_norm(self, q: np.ndarray, v: np.ndarray) -> float:
        return self.inner(q, v, v)
    
    def Exp(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        if self.use_analytic:
            return self.Q.exp(q, v)

        from Numerics.geodesic_bvp import solve_geodesic_ivp

        self._solver_diagnostics["ivp_calls"] += 1
        return solve_geodesic_ivp(self, q, v)

    def Log(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        if self.use_analytic:
            return self.Q.log(q, p)
        
        from Numerics.geodesic_bvp import solve_geodesic_bvp
        
        self._solver_diagnostics["bvp_calls"] += 1
        _, xi = solve_geodesic_bvp(self, q, p)
        return xi

    def dist(self, q: np.ndarray, p: np.ndarray) -> float:
        xi = self.Log(q, p)
        return self.norm(q, xi)

    def interpolate(self, q: np.ndarray, p: np.ndarray, u: float) -> np.ndarray:
        return self.Exp(q, float(u) * self.Log(q, p))

