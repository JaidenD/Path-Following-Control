import numpy as np
from scipy.optimize import minimize_scalar

def minimize_eta(objective, eta_guess=None, window=0.1, global_grid=80, candidates=4, closed=True):
    """
    One-dimensional closest-point search over eta.

    If eta_guess is provided, search locally. If not,
    do a coarse global scan first.
    """
    intervals = []

    # If provided with eta_guess then make an interval centered at it and search locally
    if eta_guess is not None:
        intervals.append((eta_guess - window, eta_guess + window))
    else:
        grid = np.linspace(0.0, 1.0, global_grid, endpoint=False)
        values = np.array([objective(s) for s in grid])

        # The indicies of the n smallest objective vals 
        best_indices = np.argsort(values)[:candidates]
        step = 1.0 / global_grid

        for idx in best_indices:
            center = grid[idx]
            intervals.append((center - step, center + step))

    best_eta = None
    best_value = np.inf

    # search over intervals about the n smallest objective values
    for a, b in intervals:
        result = minimize_scalar(
            lambda s: objective(normalize_eta(s, closed=closed)),
            bounds=(a, b),
            method="bounded",
        )
        if result.fun < best_value:
            best_eta = normalize_eta(result.x, closed=closed)
            best_value = result.fun

    return best_eta

# Make sure s\in[0,1]
def normalize_eta(s, closed=True):
    if closed:
        return float(s % 1.0)
    return float(np.clip(s, 0.0, 1.0))