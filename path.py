"""Path primitives for one-dimensional curves on configuration manifolds."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

from Manifolds.manifold import Manifold, RiemannianManifold


def _as_riemannian(geometry) -> RiemannianManifold:
    if isinstance(geometry, RiemannianManifold):
        return geometry
    if isinstance(geometry, Manifold):
        return RiemannianManifold(geometry)
    raise TypeError("Path geometry must be a Manifold or RiemannianManifold.")


class Path:
    """Piecewise geodesic path parameterized by normalized arclength s in [0, 1]."""

    bounds: Tuple[float, float] = (0.0, 1.0)

    def __init__(self, manifold, waypoints, closed: bool = False):
        if len(waypoints) < 2:
            raise ValueError("Need at least two waypoints.")

        self.Q = _as_riemannian(manifold)
        self.M = self.Q
        self.waypoints = [self.Q.project(q) for q in waypoints]
        self.closed = bool(closed)

        if self.closed:
            self.segment_starts = self.waypoints
            self.segment_ends = self.waypoints[1:] + self.waypoints[:1]
        else:
            self.segment_starts = self.waypoints[:-1]
            self.segment_ends = self.waypoints[1:]

        self.segment_lengths = np.array(
            [
                self.Q.dist(q0, q1)
                for q0, q1 in zip(self.segment_starts, self.segment_ends)
            ],
            dtype=float,
        )

        self.length = float(np.sum(self.segment_lengths))
        if self.length <= 0:
            raise ValueError("Paths cannot have non-positive length.")

        self.normalized_segment_lengths = self.segment_lengths / self.length
        self.cumulative = np.concatenate(
            [np.array([0.0]), np.cumsum(self.normalized_segment_lengths)]
        )
        self.cumulative[-1] = 1.0

    def _normalize_s(self, s: float) -> float:
        if self.closed:
            return float(s % 1.0)
        return float(np.clip(s, 0.0, 1.0))

    def eval(self, s: float):
        s = self._normalize_s(s)

        if s == 1.0 and not self.closed:
            return self.segment_ends[-1].copy()

        i = int(np.searchsorted(self.cumulative, s, side="right") - 1)
        i = min(max(i, 0), len(self.segment_starts) - 1)

        seg_start = self.cumulative[i]
        seg_end = self.cumulative[i + 1]
        local_s = (s - seg_start) / (seg_end - seg_start)

        return self.Q.interpolate(self.segment_starts[i], self.segment_ends[i], local_s)

    __call__ = eval

    def derivative(self, s: float, eps: float = 1e-5):
        s = self._normalize_s(s)
        p = self.eval(s)

        if not self.closed and s <= eps:
            return self.Q.Log(p, self.eval(s + eps)) / eps
        if not self.closed and s >= 1.0 - eps:
            return -self.Q.Log(p, self.eval(s - eps)) / eps

        forward = self.Q.Log(p, self.eval(s + eps)) / eps
        backward = -self.Q.Log(p, self.eval(s - eps)) / eps
        return 0.5 * (forward + backward)

    def tangent(self, s: float, eps: float = 1e-5):
        p = self.eval(s)
        v = self.derivative(s, eps=eps)
        nrm = self.Q.norm(p, v)
        if nrm < 1e-12:
            raise ValueError("Path derivative is near zero; path is not regular here.")
        return v / nrm


class ParametricPath:
    """Path from user-supplied gamma(s) and optional gamma_prime(s)."""

    bounds: Tuple[float, float] = (0.0, 1.0)

    def __init__(
        self,
        manifold,
        eval_fn: Callable[[float], np.ndarray],
        derivative_fn: Callable[[float], np.ndarray] | None = None,
        *,
        closed: bool = False,
    ):
        self.Q = _as_riemannian(manifold)
        self.M = self.Q
        self.eval_fn = eval_fn
        self.derivative_fn = derivative_fn
        self.closed = bool(closed)

    def _normalize_s(self, s: float) -> float:
        if self.closed:
            return float(s % 1.0)
        return float(np.clip(s, 0.0, 1.0))

    def eval(self, s: float):
        return self.Q.project(self.eval_fn(self._normalize_s(s)))

    __call__ = eval

    def derivative(self, s: float, eps: float = 1e-5):
        s = self._normalize_s(s)
        if self.derivative_fn is not None:
            return np.asarray(self.derivative_fn(s), dtype=float).reshape(self.Q.dim)

        p = self.eval(s)
        if not self.closed and s <= eps:
            return self.Q.Log(p, self.eval(s + eps)) / eps
        if not self.closed and s >= 1.0 - eps:
            return -self.Q.Log(p, self.eval(s - eps)) / eps

        forward = self.Q.Log(p, self.eval(s + eps)) / eps
        backward = -self.Q.Log(p, self.eval(s - eps)) / eps
        return 0.5 * (forward + backward)

    def tangent(self, s: float, eps: float = 1e-5):
        p = self.eval(s)
        v = self.derivative(s, eps=eps)
        nrm = self.Q.norm(p, v)
        if nrm < 1e-12:
            raise ValueError("Path derivative is near zero; path is not regular here.")
        return v / nrm


PiecewiseGeodesicPath = Path


class CubicSplinePath:
    """Smooth coordinate cubic spline projected back onto the manifold.

    This is the pragmatic spline for robot joint-space testing. It unwraps
    waypoint coordinates using the manifold logarithm, fits a SciPy
    ``CubicSpline`` in that local coordinate chart, and projects evaluated
    points back onto the manifold. For products of Euclidean, prismatic, and
    revolute joints this gives the expected smooth joint-space path.

    It is not a Riemannian cubic spline solver. If the metric is strongly
    curved or the path crosses chart singularities, prefer an explicit
    ``ParametricPath`` or a specialized path construction.
    """

    bounds: Tuple[float, float] = (0.0, 1.0)

    def __init__(
        self,
        manifold,
        waypoints,
        *,
        knots=None,
        closed: bool = False,
        bc_type: str = "natural",
    ):
        if len(waypoints) < 2:
            raise ValueError("Need at least two waypoints.")

        self.Q = _as_riemannian(manifold)
        self.M = self.Q
        self.closed = bool(closed)
        self.bc_type = bc_type
        self.waypoints = [self.Q.project(q) for q in waypoints]

        spline_points = list(self.waypoints)
        if self.closed:
            spline_points.append(self.waypoints[0])

        self._chart_points = self._unwrap_points(spline_points)

        if knots is None:
            distances = [
                self.Q.dist(q0, q1)
                for q0, q1 in zip(spline_points[:-1], spline_points[1:])
            ]
            cumulative = np.concatenate([[0.0], np.cumsum(distances)])
            if cumulative[-1] <= 0.0:
                raise ValueError("Spline path cannot have non-positive length.")
            self.knots = cumulative / cumulative[-1]
        else:
            self.knots = np.asarray(knots, dtype=float).reshape(-1)
            if self.closed and len(self.knots) == len(self.waypoints):
                self.knots = np.concatenate([self.knots, [1.0]])

        if self.knots.shape != (len(self._chart_points),):
            raise ValueError(
                f"Expected {len(self._chart_points)} knots, got {len(self.knots)}."
            )
        if not np.all(np.diff(self.knots) > 0.0):
            raise ValueError("Spline knots must be strictly increasing.")
        if abs(self.knots[0]) > 1e-12 or abs(self.knots[-1] - 1.0) > 1e-12:
            raise ValueError("Spline knots must start at 0 and end at 1.")

        if bc_type == "periodic" and not np.allclose(
            self._chart_points[0], self._chart_points[-1], atol=1e-10
        ):
            raise ValueError(
                "Periodic cubic splines require matching first/last chart coordinates. "
                "Use bc_type='natural' for wrapped or winding paths."
            )

        self.spline = CubicSpline(self.knots, self._chart_points, axis=0, bc_type=bc_type)

    def _unwrap_points(self, points):
        chart_points = [self.Q.coordinate(points[0])]
        for prev, current in zip(points[:-1], points[1:]):
            chart_points.append(chart_points[-1] + self.Q.Q.log(prev, current))
        return np.asarray(chart_points, dtype=float)

    def _normalize_s(self, s: float) -> float:
        if self.closed:
            return float(s % 1.0)
        return float(np.clip(s, 0.0, 1.0))

    def eval(self, s: float):
        chart_q = np.asarray(self.spline(self._normalize_s(s)), dtype=float).reshape(self.Q.dim)
        return self.Q.project(chart_q)

    __call__ = eval

    def derivative(self, s: float, eps: float = 1e-5):
        return np.asarray(self.spline(self._normalize_s(s), 1), dtype=float).reshape(self.Q.dim)

    def tangent(self, s: float, eps: float = 1e-5):
        p = self.eval(s)
        v = self.derivative(s, eps=eps)
        nrm = self.Q.norm(p, v)
        if nrm < 1e-12:
            raise ValueError("Path derivative is near zero; path is not regular here.")
        return v / nrm


def coordinate_circle_path(
    manifold,
    *,
    center=None,
    radius: float = 0.8,
    axes=(0, 1),
    closed: bool = True,
) -> ParametricPath:
    """A simple coordinate-space circle, projected onto the manifold."""
    Q = _as_riemannian(manifold)
    if Q.dim < 2:
        raise ValueError("coordinate_circle_path needs a manifold with dim >= 2.")
    center = np.zeros(Q.dim) if center is None else np.asarray(center, dtype=float).reshape(Q.dim)
    i, j = axes

    def gamma(s):
        angle = 2.0 * np.pi * s
        q = center.copy()
        q[i] += radius * np.sin(angle)
        q[j] += radius * np.cos(angle)
        return q

    def gamma_prime(s):
        angle = 2.0 * np.pi * s
        dq = np.zeros(Q.dim)
        dq[i] = radius * 2.0 * np.pi * np.cos(angle)
        dq[j] = -radius * 2.0 * np.pi * np.sin(angle)
        return dq

    return ParametricPath(Q, gamma, gamma_prime, closed=closed)
