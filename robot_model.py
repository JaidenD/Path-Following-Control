from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

from Manifolds.manifold import Manifold


class RobotModel(ABC):
    """Minimal fully-actuated second-order robot model interface."""

    def __init__(self, dim: int, manifold: Optional[Manifold] = None):
        if dim <= 0:
            raise ValueError("Robot dimension must be positive.")
        self.dim = int(dim)
        self.manifold = manifold

    def project_configuration(self, q):
        q = np.asarray(q, dtype=float).reshape(self.dim)
        if self.manifold is None:
            return q
        return self.manifold.project(q)

    @abstractmethod
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Return M(q)."""

    def bias(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """Return C(q,qdot)qdot + G(q), or another model bias term."""
        return np.zeros(self.dim)

    def forward_dynamics(self, q, qdot, u):
        q = self.project_configuration(q)
        qdot = np.asarray(qdot, dtype=float).reshape(self.dim)
        u = np.asarray(u, dtype=float).reshape(self.dim)
        return np.linalg.solve(self.mass_matrix(q), u - self.bias(q, qdot))

    def inverse_dynamics(self, q, qdot, qddot_cmd):
        q = self.project_configuration(q)
        qdot = np.asarray(qdot, dtype=float).reshape(self.dim)
        qddot_cmd = np.asarray(qddot_cmd, dtype=float).reshape(self.dim)
        return self.mass_matrix(q) @ qddot_cmd + self.bias(q, qdot)

    def state_dot(self, q, qdot, u):
        qdot = np.asarray(qdot, dtype=float).reshape(self.dim)
        qddot = self.forward_dynamics(q, qdot, u)
        return qdot, qddot

    def wrap_state(self, state):
        q, qdot = split_state(state, self.dim)
        return np.column_stack((self.project_configuration(q), qdot))


class FullyActuatedRobotModel(RobotModel):
    """RobotModel backed by callables for M(q) and bias(q, qdot)."""

    def __init__(
        self,
        dim: int,
        mass_matrix_fn: Callable[[np.ndarray], np.ndarray],
        bias_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
        *,
        manifold: Optional[Manifold] = None,
    ):
        super().__init__(dim, manifold=manifold)
        self.mass_matrix_fn = mass_matrix_fn
        self.bias_fn = bias_fn

    def mass_matrix(self, q):
        M = np.asarray(self.mass_matrix_fn(self.project_configuration(q)), dtype=float)
        if M.shape != (self.dim, self.dim):
            raise ValueError(f"Mass matrix shape {M.shape}; expected ({self.dim}, {self.dim}).")
        return M

    def bias(self, q, qdot):
        if self.bias_fn is None:
            return np.zeros(self.dim)
        q = self.project_configuration(q)
        qdot = np.asarray(qdot, dtype=float).reshape(self.dim)
        b = np.asarray(self.bias_fn(q, qdot), dtype=float).reshape(self.dim)
        return b


def split_state(state, dim: int):
    """Accept either a 2-column [q, qdot] state or a flat [q, qdot] vector."""
    state = np.asarray(state, dtype=float)
    if state.shape == (dim, 2):
        return state[:, 0], state[:, 1]
    if state.shape == (2 * dim,):
        return state[:dim], state[dim:]
    raise ValueError(f"Expected state shape ({dim}, 2) or ({2 * dim},), got {state.shape}.")
