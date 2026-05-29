from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Union

import numpy as np

from Numerics.closest_point import ClosestPointResult, closest_point
from robot_model import RobotModel, split_state


ScalarReference = Union[float, Callable[[float], float], Callable[[float, float], float]]


@dataclass
class PathFollowingCoordinates:
    eta: float
    point: np.ndarray
    xi: np.ndarray
    distance: float
    closest: ClosestPointResult


@dataclass
class PathFollowingOutput:
    u: np.ndarray
    eta: float
    xi: np.ndarray
    qdot_des: np.ndarray
    qddot_cmd: np.ndarray
    coordinates: PathFollowingCoordinates


class EtaController(Protocol):
    def command(self, eta: float, path, Q, t: float = 0.0) -> np.ndarray:
        ...


class XiController(Protocol):
    def command(self, coordinates: PathFollowingCoordinates, Q, t: float = 0.0) -> np.ndarray:
        ...


class EtaXiTransform:
    """Metric closest-point transform q -> (eta, xi)."""

    def __init__(self, Q, path, *, window: float = 0.1, global_grid: int = 80):
        self.Q = Q
        self.path = path
        self.window = float(window)
        self.global_grid = int(global_grid)

    def __call__(self, q, eta_guess: Optional[float] = None) -> PathFollowingCoordinates:
        result = closest_point(
            self.Q,
            q,
            self.path,
            eta_guess=eta_guess,
            window=self.window,
            global_grid=self.global_grid,
        )
        return PathFollowingCoordinates(
            eta=result.eta,
            point=result.point,
            xi=result.xi,
            distance=result.distance,
            closest=result,
        )


def _eval_reference(ref: ScalarReference, eta: float, t: float) -> float:
    if callable(ref):
        try:
            return float(ref(t, eta))
        except TypeError:
            return float(ref(t))
    return float(ref)


@dataclass
class EtaVelocityController:
    """Track a desired metric speed along the path tangent."""

    speed: ScalarReference

    def command(self, eta: float, path, Q, t: float = 0.0) -> np.ndarray:
        speed = _eval_reference(self.speed, eta, t)
        return speed * path.tangent(eta)


@dataclass
class EtaPositionController:
    """Track a desired scalar path parameter eta_ref."""

    eta_ref: ScalarReference
    kp: float
    feedforward_eta_dot: ScalarReference = 0.0
    closed: bool = False

    def command(self, eta: float, path, Q, t: float = 0.0) -> np.ndarray:
        ref = _eval_reference(self.eta_ref, eta, t)
        error = ref - eta
        if self.closed:
            error = (error + 0.5) % 1.0 - 0.5
        eta_dot = _eval_reference(self.feedforward_eta_dot, eta, t) + self.kp * error
        return eta_dot * path.derivative(eta)


@dataclass
class XiStabilizer:
    gain: float

    def command(self, coordinates: PathFollowingCoordinates, Q, t: float = 0.0) -> np.ndarray:
        return -self.gain * coordinates.xi


class ComputedTorquePathFollowingController:
    """Map eta/xi velocity commands to torques for fully actuated robots."""

    def __init__(
        self,
        Q,
        path,
        robot: RobotModel,
        eta_controller: EtaController,
        xi_controller: Optional[XiController] = None,
        *,
        damping_gain: float = 6.0,
        transform: Optional[EtaXiTransform] = None,
    ):
        if Q.dim != robot.dim:
            raise ValueError(f"Q dimension {Q.dim} does not match robot dimension {robot.dim}.")
        self.Q = Q
        self.path = path
        self.robot = robot
        self.eta_controller = eta_controller
        self.xi_controller = xi_controller if xi_controller is not None else XiStabilizer(0.0)
        self.damping_gain = float(damping_gain)
        self.transform = transform if transform is not None else EtaXiTransform(Q, path)

    def command(self, q, qdot, *, t: float = 0.0, eta_guess: Optional[float] = None):
        q = self.Q.project(q)
        qdot = np.asarray(qdot, dtype=float).reshape(self.Q.dim)

        coordinates = self.transform(q, eta_guess=eta_guess)
        eta_velocity = self.eta_controller.command(coordinates.eta, self.path, self.Q, t=t)
        xi_velocity = self.xi_controller.command(coordinates, self.Q, t=t)
        qdot_des = np.asarray(eta_velocity + xi_velocity, dtype=float).reshape(self.Q.dim)

        qddot_cmd = -self.damping_gain * (qdot - qdot_des)
        u = self.robot.inverse_dynamics(q, qdot, qddot_cmd)

        return PathFollowingOutput(
            u=u,
            eta=coordinates.eta,
            xi=coordinates.xi,
            qdot_des=qdot_des,
            qddot_cmd=qddot_cmd,
            coordinates=coordinates,
        )

    def command_from_state(self, state, *, t: float = 0.0, eta_guess: Optional[float] = None):
        q, qdot = split_state(state, self.robot.dim)
        return self.command(q, qdot, t=t, eta_guess=eta_guess)
