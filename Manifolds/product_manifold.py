import numpy as np

from Manifolds.manifold import Manifold


class ProductManifold(Manifold):
    """Cartesian product of coordinate manifolds."""

    def __init__(self, *factors: Manifold):
        if len(factors) < 1:
            raise ValueError("ProductManifold needs at least one factor.")

        self.factors = tuple(factor() if isinstance(factor, type) else factor for factor in factors)
        self.dim = int(sum(factor.dim for factor in self.factors))

    def _split(self, q):
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape != (self.dim,):
            raise ValueError(f"Expected point with shape ({self.dim},), got {q.shape}.")

        parts = []
        start = 0
        for factor in self.factors:
            stop = start + factor.dim
            parts.append(q[start:stop])
            start = stop
        return parts

    def _join(self, parts):
        return np.concatenate([np.asarray(part, dtype=float).reshape(-1) for part in parts])

    def coordinate(self, q):
        return self._join(
            factor.coordinate(part) for factor, part in zip(self.factors, self._split(q))
        )

    def project(self, q):
        return self._join(
            factor.project(part) for factor, part in zip(self.factors, self._split(q))
        )

    def exp(self, q, v):
        q_parts = self._split(q)
        v_parts = self._split(v)
        return self._join(
            factor.exp(q_part, v_part)
            for factor, q_part, v_part in zip(self.factors, q_parts, v_parts)
        )

    def log(self, q, p):
        q_parts = self._split(q)
        p_parts = self._split(p)
        return self._join(
            factor.log(q_part, p_part)
            for factor, q_part, p_part in zip(self.factors, q_parts, p_parts)
        )

    def interpolate(self, q, p, u):
        return self.exp(q, float(u) * self.log(q, p))
