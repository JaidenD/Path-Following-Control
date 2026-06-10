from __future__ import annotations

import numpy as np

from Manifolds.manifold import Manifold


def wrap_angle(theta):
    """Project angles to principle branch (-pi, pi]."""
    return (np.asarray(theta, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


class Prismatic(Manifold):
    """One-dimensional prismatic joint.

    If bounds are provided, ``project`` clips to them. Without bounds this is
    simply R^1.
    """

    def __init__(self, lower: float | None = None, upper: float | None = None):
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError("Prismatic lower bound must be less than upper bound.")
        self.dim = 1
        self.lower = lower
        self.upper = upper

    def coordinate(self, q):
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape != (1,):
            raise ValueError(f"Expected scalar prismatic coordinate, got shape {q.shape}.")
        return q

    def project(self, q):
        q = self.coordinate(q)
        if self.lower is not None or self.upper is not None:
            return np.clip(q, self.lower, self.upper)
        return q


class Revolute(Manifold):
    """Circle manifold S^1 represented by a wrapped angle."""

    def __init__(self):
        self.dim = 1

    def coordinate(self, theta):
        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.shape != (1,):
            raise ValueError(f"Expected scalar revolute coordinate, got shape {theta.shape}.")
        return theta

    def project(self, theta):
        return wrap_angle(theta).reshape(1)

    def exp(self, q, v):
        return self.project(self.coordinate(q) + np.asarray(v, dtype=float).reshape(1))

    def log(self, q, p):
        return wrap_angle(self.coordinate(p) - self.coordinate(q)).reshape(1)

    def dist(self, q, p):
        return float(abs(self.log(q, p)[0]))


# Backwards-compatible misspelling from the prototype.
Revoloute = Revolute
