from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class ClosestPointResult:
    eta: float
    point: np.ndarray
    xi: np.ndarray
    distance: float
    objective: float
    success: bool
    message: str = ""
    normality_residual: float | None = None


def _path_eval(path, s: float):
    return path.eval(s)


def closest_point(
    Q,
    q,
    path,
    *,
    eta_guess: Optional[float] = None,
    window: float = 0.1,
    global_grid: int = 80,
    candidates: int = 4,
) -> ClosestPointResult:
    """Find the closest point on a 1D path under the supplied metric.

    The objective is ``||Log_{path(s)}(q)||_G^2``. A local window around
    ``eta_guess`` is used when available; otherwise a coarse global scan seeds a
    few bounded scalar optimizations.
    """
    q = Q.project(q)
    closed = bool(getattr(path, "closed", False))

    def normalize_s(s: float) -> float:
        if closed:
            return float(s % 1.0)
        return float(np.clip(s, 0.0, 1.0))

    def objective_unwrapped(s: float) -> float:
        eta = normalize_s(s)
        p = _path_eval(path, eta)
        try:
            xi = Q.Log(p, q)
        except RuntimeError:
            return float("inf")
        return Q.squared_norm(p, xi)

    intervals = []
    if eta_guess is not None:
        center = float(eta_guess)
        if closed:
            intervals.append((center - window, center + window))
        else:
            intervals.append((max(0.0, center - window), min(1.0, center + window)))
    else:
        endpoint = not closed
        grid = np.linspace(0.0, 1.0, max(global_grid, 3), endpoint=endpoint)
        values = np.array([objective_unwrapped(s) for s in grid])
        finite = np.isfinite(values)
        if not np.any(finite):
            raise RuntimeError("Closest-point search failed: all objective values are invalid.")

        order = np.argsort(values[finite])
        finite_indices = np.nonzero(finite)[0][order[: max(1, candidates)]]
        step = 1.0 / (len(grid) if closed else len(grid) - 1)

        for idx in finite_indices:
            center = float(grid[idx])
            if closed:
                intervals.append((center - step, center + step))
            else:
                intervals.append((max(0.0, center - step), min(1.0, center + step)))

    best = None
    for bounds in intervals:
        if bounds[1] <= bounds[0]:
            continue

        result = minimize_scalar(objective_unwrapped, bounds=bounds, method="bounded")
        candidate = (result.fun, result.x, result)
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        raise RuntimeError("Closest-point search failed: no valid optimization interval.")

    objective, eta_raw, opt = best
    eta = normalize_s(eta_raw)
    p = _path_eval(path, eta)
    xi = Q.Log(p, q)
    objective = Q.squared_norm(p, xi)
    distance = float(np.sqrt(max(objective, 0.0)))

    normality_residual = None
    if hasattr(path, "derivative"):
        tangent = path.derivative(eta)
        denom = Q.norm(p, xi) * Q.norm(p, tangent)
        if denom > 1e-12:
            normality_residual = Q.inner(p, xi, tangent) / denom

    return ClosestPointResult(
        eta=eta,
        point=p,
        xi=xi,
        distance=distance,
        objective=float(objective),
        success=bool(opt.success),
        message=str(opt.message),
        normality_residual=normality_residual,
    )
