import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Manifolds.joints import Revolute
from Manifolds.manifold import RiemannianManifold
from Manifolds.product_manifold import ProductManifold
from Numerics.closest_point import closest_point
from decoupled_controllers import (
    ComputedTorquePathFollowingController,
    EtaVelocityController,
    EtaXiTransform,
    XiStabilizer,
)
from path import (
    CubicSplinePath,
    coordinate_circle_path,
    coordinate_figure_eight_path,
    coordinate_lissajous_path,
)
from robot_model import FullyActuatedRobotModel


# ============================================================
# Configuration manifold: Q = S^1 x S^1
# ============================================================

CONFIG_MANIFOLD = ProductManifold(Revolute(), Revolute())
Q = RiemannianManifold(CONFIG_MANIFOLD, np.eye(2))

def wrap(theta):
    """
    Wrap angles to (-pi, pi].
    Works for scalars or numpy arrays.
    """
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def T2_exp(q, v):
    """
    Exp_q(v) on S^1 x S^1 with standard flat product metric.

    q, v are both shape (2,).
    """
    return Q.Exp(q, v)


def T2_log(q, p):
    """
    Log_q(p) on S^1 x S^1 with standard flat product metric.

    q, p are both shape (2,).
    Returns the wrapped displacement from q to p.
    """
    return Q.Log(q, p)


def T2_dist(q, p):
    """
    Distance on S^1 x S^1 with standard flat product metric.
    """
    return Q.dist(q, p)


# ============================================================
# Physical parameters
# ============================================================

params = {
    "m1": 1.0,
    "m2": 1.0,
    "l1": 1.0,
    "l2": 1.0,
    "lc1": 0.5,
    "lc2": 0.5,
    "I1": 0.1,
    "I2": 0.1,
    "g": 9.81,
}


# ============================================================
# Two-link manipulator dynamics
# ============================================================

def mass_matrix(q):
    """
    Return M(q) for a standard planar 2R manipulator.

    q[0] = theta1
    q[1] = theta2, relative angle of link 2 wrt link 1
    """
    theta2 = q[1]

    m1 = params["m1"]
    m2 = params["m2"]
    l1 = params["l1"]
    lc1 = params["lc1"]
    lc2 = params["lc2"]
    I1 = params["I1"]
    I2 = params["I2"]

    c2 = np.cos(theta2)

    M11 = I1 + I2 + m1 * lc1**2 + m2 * (
        l1**2 + lc2**2 + 2.0 * l1 * lc2 * c2
    )
    M12 = I2 + m2 * (lc2**2 + l1 * lc2 * c2)
    M22 = I2 + m2 * lc2**2

    return np.array([
        [M11, M12],
        [M12, M22],
    ])


def coriolis_vector(q, qdot):
    """
    Return C(q, qdot) qdot as a vector.
    """
    theta2 = q[1]
    dtheta1, dtheta2 = qdot

    m2 = params["m2"]
    l1 = params["l1"]
    lc2 = params["lc2"]

    h = -m2 * l1 * lc2 * np.sin(theta2)

    C1 = h * (2.0 * dtheta1 * dtheta2 + dtheta2**2)
    C2 = -h * dtheta1**2

    return np.array([C1, C2])


def gravity_vector(q):
    """
    Return G(q).

    Convention:
    theta1 is absolute angle of link 1.
    theta2 is relative angle of link 2 wrt link 1.
    """
    theta1, theta2 = q

    m1 = params["m1"]
    m2 = params["m2"]
    l1 = params["l1"]
    lc1 = params["lc1"]
    lc2 = params["lc2"]
    g = params["g"]

    G1 = (m1 * lc1 + m2 * l1) * g * np.cos(theta1) \
         + m2 * lc2 * g * np.cos(theta1 + theta2)

    G2 = m2 * lc2 * g * np.cos(theta1 + theta2)

    return np.array([G1, G2])


def forward_dynamics(q, qdot, tau):
    """
    Solve

        M(q) qddot + C(q,qdot)qdot + G(q) = tau

    for qddot.
    """
    M = mass_matrix(q)
    C = coriolis_vector(q, qdot)
    G = gravity_vector(q)

    return np.linalg.solve(M, tau - C - G)


def state_dot(state, tau):
    """
    State convention:

        state[:, 0] = q
        state[:, 1] = qdot

    For the two-link manipulator:

        state =
        [[theta1, dtheta1],
         [theta2, dtheta2]]
    """
    q = state[:, 0]
    qdot = state[:, 1]

    qddot = forward_dynamics(q, qdot, tau)

    return np.column_stack((qdot, qddot))


def wrap_state(state):
    """
    Wrap the configuration q back onto S^1 x S^1.
    Do not wrap qdot.
    """
    q = state[:, 0]
    qdot = state[:, 1]

    q_wrapped = T2_exp(np.zeros(2), q)

    return np.column_stack((q_wrapped, qdot))


# ============================================================
# Example paths gamma : [0, 1] -> S^1 x S^1
# ============================================================

PATH_KIND = os.environ.get("PATH_KIND", "circle")


def make_spline_test_path():
    waypoints = [
        np.array([0.0, 0.75]),
        np.array([0.65, 0.2]),
        np.array([0.35, -0.65]),
        np.array([-0.45, -0.45]),
        np.array([-0.75, 0.25]),
    ]
    return CubicSplinePath(Q, waypoints, closed=True, bc_type="periodic")


def make_configuration_path(kind=PATH_KIND):
    if kind == "circle":
        return coordinate_circle_path(Q, radius=0.8)
    if kind == "figure_eight":
        return coordinate_figure_eight_path(Q, amplitudes=(0.8, 0.45))
    if kind == "lissajous":
        return coordinate_lissajous_path(
            Q,
            amplitudes=np.array([0.7, 0.5]),
            frequencies=np.array([1.0, 2.0]),
            phases=np.array([0.0, np.pi / 3.0]),
        )
    if kind == "spline":
        return make_spline_test_path()
    raise ValueError(f"Unknown path kind {kind!r}.")


configuration_path = make_configuration_path(PATH_KIND)


def gamma(s):
    """Evaluate the currently selected test path."""
    return configuration_path.eval(s)


def gamma_prime(s):
    """Derivative of the currently selected test path."""
    return configuration_path.derivative(s)


def tangent_unit(s):
    """
    Unit tangent vector to gamma at s.
    """
    return configuration_path.tangent(s)


def normal_unit(s):
    """
    Unit normal vector to gamma at s.

    Since Q is 2D and the path is 1D, the normal space is 1D.
    """
    T = tangent_unit(s)

    return np.array([
        -T[1],
        T[0],
    ])


# ============================================================
# eta-xi coordinate transform
# ============================================================

def displacement_from_path(s, q):
    """
    e(s, q) = Log_{gamma(s)}(q).
    """
    p = gamma(s)
    return Q.Log(p, q)


def squared_distance_to_path(s, q):
    """
    Squared distance from q to gamma(s).
    """
    p = gamma(s)
    e = displacement_from_path(s, q)
    return Q.squared_norm(p, e)


def closest_eta(q, eta_guess=None, window=0.1):
    """
    Find eta = argmin_s dist(gamma(s), q)^2.

    If eta_guess is None:
        search globally on [0, 1].

    If eta_guess is provided:
        search locally around eta_guess.
    """

    result = closest_point(
        Q,
        q,
        configuration_path,
        eta_guess=eta_guess,
        window=window,
    )
    return result.eta


def eta_xi_transform(q, eta_guess=None, *, return_vector=False):
    """
    Compute q -> (eta, xi) for q in S^1 x S^1.

    eta:
        closest path parameter

    xi:
        scalar transverse displacement in the normal direction
    """
    result = closest_point(Q, q, configuration_path, eta_guess=eta_guess)

    if return_vector:
        return result.eta, result.xi

    N = normal_unit(result.eta)
    xi = N @ result.xi

    return result.eta, xi
def estimate_path_tube_radius():
    """
    Rough visualization radius for the normal tube.

    For the default circle this is the true focal radius. For the other test
    paths it is only a conservative plotting radius.
    """
    exp_inj_radius = np.pi
    path_focal_radius = 0.8 if PATH_KIND == "circle" else 0.3

    return min(exp_inj_radius, path_focal_radius)


def tube_boundary_points(epsilon, n=600):
    """
    Build boundary curves of the normal tube around gamma.

    Returns:
        upper_boundary, lower_boundary

    where
        upper = gamma(s) + epsilon N(s)
        lower = gamma(s) - epsilon N(s)
    """
    s_grid = np.linspace(0.0, 1.0, n)

    upper = []
    lower = []

    for s in s_grid:
        p = gamma(s)
        N = normal_unit(s)

        upper.append(T2_exp(p, epsilon * N))
        lower.append(T2_exp(p, -epsilon * N))

    return np.array(upper), np.array(lower)

# ============================================================
# Path-following controller
# ============================================================

controller_params = {
    "v_eta": 0.25,   # desired speed along the path direction
    "k_xi": 4.0,     # transverse correction gain
    "k_d": 6.0,      # velocity damping gain
}


two_link_robot = FullyActuatedRobotModel(
    dim=2,
    mass_matrix_fn=mass_matrix,
    bias_fn=lambda q, qdot: coriolis_vector(q, qdot) + gravity_vector(q),
    manifold=CONFIG_MANIFOLD,
)

path_controller = ComputedTorquePathFollowingController(
    Q,
    configuration_path,
    two_link_robot,
    eta_controller=EtaVelocityController(speed=controller_params["v_eta"]),
    xi_controller=XiStabilizer(gain=controller_params["k_xi"]),
    damping_gain=controller_params["k_d"],
    transform=EtaXiTransform(Q, configuration_path, window=0.1),
)


def path_following_controller(state, eta_guess=None):
    """
    Simple path-following controller for the 2-link manipulator.

    Coordinate transform:

        q -> (eta, xi)

    Desired velocity field:

        qdot_des = v_eta T(eta) - k_xi xi N(eta)

    Computed torque form:

        tau = M(q) qddot_cmd + C(q,qdot)qdot + G(q)

    where

        qddot_cmd = -k_d (qdot - qdot_des)

    Inputs:
        state[:, 0] = q
        state[:, 1] = qdot

    Returns:
        tau, eta, xi
    """
    q = state[:, 0]
    qdot = state[:, 1]

    output = path_controller.command(q, qdot, eta_guess=eta_guess)
    xi_scalar = normal_unit(output.eta) @ output.xi

    return output.u, output.eta, xi_scalar


# ============================================================
# Numerical integration
# ============================================================

def euler_step(state, tau, dt):
    """
    One explicit Euler step.
    """
    next_state = state + dt * state_dot(state, tau)
    return wrap_state(next_state)


def rk4_step(state, tau_func, dt):
    """
    One RK4 step.

    tau_func takes state and returns tau.
    """

    def f(x):
        tau = tau_func(x)
        return state_dot(x, tau)

    k1 = f(state)
    k2 = f(wrap_state(state + 0.5 * dt * k1))
    k3 = f(wrap_state(state + 0.5 * dt * k2))
    k4 = f(wrap_state(state + dt * k3))

    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return wrap_state(next_state)


def simulate_path_following(state0, dt=0.001, T_final=5.0, use_rk4=True):
    """
    Simulate closed-loop path following.

    Returns a dictionary containing histories of:
        t, q, qdot, eta, xi, tau
    """
    n_steps = int(T_final / dt)

    state = state0.copy()
    eta_guess = None

    history = {
        "t": [],
        "q": [],
        "qdot": [],
        "eta": [],
        "xi": [],
        "tau": [],
    }

    for k in range(n_steps):
        t = k * dt

        tau, eta, xi = path_following_controller(state, eta_guess=eta_guess)
        eta_guess = eta

        history["t"].append(t)
        history["q"].append(state[:, 0].copy())
        history["qdot"].append(state[:, 1].copy())
        history["eta"].append(eta)
        history["xi"].append(xi)
        history["tau"].append(tau.copy())

        if use_rk4:
            def tau_func(x):
                tau_x, _, _ = path_following_controller(x, eta_guess=eta_guess)
                return tau_x

            state = rk4_step(state, tau_func, dt)
        else:
            state = euler_step(state, tau, dt)

    for key in history:
        history[key] = np.array(history[key])

    return history


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # Basic dynamics test
    # --------------------------------------------------------

    q = np.zeros(2)
    qdot = np.zeros(2)

    state = np.column_stack((q, qdot))

    tau = np.zeros(2)

    xdot = state_dot(state, tau)

    print("state =")
    print(state)

    print("\nxdot =")
    print(xdot)

    print("\nM(q) =")
    print(mass_matrix(q))

    print("\nC(q,qdot)qdot =")
    print(coriolis_vector(q, qdot))

    print("\nG(q) =")
    print(gravity_vector(q))

    # --------------------------------------------------------
    # Test S1 x S1 geometry
    # --------------------------------------------------------

    p = np.array([np.pi / 2, -np.pi / 2])
    v = T2_log(q, p)

    print("\nT2_log(q, p) =")
    print(v)

    print("\nT2_exp(q, T2_log(q,p)) =")
    print(T2_exp(q, v))

    print("\nT2_dist(q, p) =")
    print(T2_dist(q, p))

    # --------------------------------------------------------
    # Test eta-xi coordinates
    # --------------------------------------------------------

    q_test = np.array([0.2, 0.7])

    eta, xi = eta_xi_transform(q_test)

    print("\nq_test =")
    print(q_test)

    print("\neta =")
    print(eta)

    print("\nxi =")
    print(xi)

    print("\ngamma(eta) =")
    print(gamma(eta))

    print("\nLog_gamma(eta)(q_test) =")
    print(displacement_from_path(eta, q_test))

    # --------------------------------------------------------
    # Test controller
    # --------------------------------------------------------

    q0 = np.array([0.2, 0.7])
    qdot0 = np.array([0.0, 0.0])
    state0 = np.column_stack((q0, qdot0))

    tau0, eta0, xi0 = path_following_controller(state0)

    print("\nController test")
    print("tau0 =")
    print(tau0)

    print("\neta0 =")
    print(eta0)

    print("\nxi0 =")
    print(xi0)

    # --------------------------------------------------------
    # Simulate closed-loop path following
    # --------------------------------------------------------

    hist = simulate_path_following(
        state0,
        dt=0.001,
        T_final=5.0,
        use_rk4=True,
    )

    print("\nSimulation complete.")

    print("\nFinal q =")
    print(hist["q"][-1])

    print("\nFinal qdot =")
    print(hist["qdot"][-1])

    print("\nFinal eta =")
    print(hist["eta"][-1])

    print("\nFinal xi =")
    print(hist["xi"][-1])

    print("\nFinal tau =")
    print(hist["tau"][-1])

    # --------------------------------------------------------
    # One large summary plot
    # --------------------------------------------------------

    import matplotlib.pyplot as plt

    t = hist["t"]
    q_hist = hist["q"]
    qdot_hist = hist["qdot"]
    eta_hist = hist["eta"]
    xi_hist = hist["xi"]
    tau_hist = hist["tau"]

    gamma_hist = np.array([gamma(eta) for eta in eta_hist])

    e_hist = np.array([
        T2_log(gamma_hist[k], q_hist[k])
        for k in range(len(t))
    ])
    e_norm = np.linalg.norm(e_hist, axis=1)

    s_grid = np.linspace(0.0, 1.0, 400)
    path_points = np.array([gamma(s) for s in s_grid])

    fig, axs = plt.subplots(3, 2, figsize=(14, 12))

    # --------------------------------------------------------
    # 1. Path in configuration space
    # --------------------------------------------------------

    tube_radius = estimate_path_tube_radius()
    tube_radius_plot = 0.95 * tube_radius

    tube_upper, tube_lower = tube_boundary_points(tube_radius_plot)

    # Build a closed polygon for the tube region
    tube_polygon = np.vstack([
        tube_upper,
        tube_lower[::-1],
        tube_upper[0:1],
    ])

    axs[0, 0].fill(
        tube_polygon[:, 0],
        tube_polygon[:, 1],
        alpha=0.2,
        color="tab:blue",
        label=rf"normal tube, $\varepsilon \approx {tube_radius_plot:.2f}$",
    )

    axs[0, 0].plot(path_points[:, 0], path_points[:, 1], label=r"$\gamma(s)$")
    axs[0, 0].plot(q_hist[:, 0], q_hist[:, 1], label=r"$q(t)$")
    axs[0, 0].scatter(q_hist[0, 0], q_hist[0, 1], marker="o", label="start")
    axs[0, 0].scatter(q_hist[-1, 0], q_hist[-1, 1], marker="x", label="end")

    axs[0, 0].set_xlabel(r"$\theta_1$")
    axs[0, 0].set_ylabel(r"$\theta_2$")
    axs[0, 0].set_title(
        rf"Path following on $S^1 \times S^1$; "
        rf"$\mathrm{{inj}}(\exp)=\pi$, tube radius $\approx {tube_radius:.2f}$"
    )
    axs[0, 0].axis("equal")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # --------------------------------------------------------
    # 2. Joint angles
    # --------------------------------------------------------

    axs[0, 1].plot(t, q_hist[:, 0], label=r"$\theta_1$")
    axs[0, 1].plot(t, q_hist[:, 1], label=r"$\theta_2$")
    axs[0, 1].set_xlabel("time [s]")
    axs[0, 1].set_ylabel("angle [rad]")
    axs[0, 1].set_title("Joint angles")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # --------------------------------------------------------
    # 3. Joint velocities
    # --------------------------------------------------------

    axs[1, 0].plot(t, qdot_hist[:, 0], label=r"$\dot{\theta}_1$")
    axs[1, 0].plot(t, qdot_hist[:, 1], label=r"$\dot{\theta}_2$")
    axs[1, 0].set_xlabel("time [s]")
    axs[1, 0].set_ylabel("velocity [rad/s]")
    axs[1, 0].set_title("Joint velocities")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # --------------------------------------------------------
    # 4. eta and xi
    # --------------------------------------------------------

    axs[1, 1].plot(t, eta_hist, label=r"$\eta$")
    axs[1, 1].plot(t, xi_hist, label=r"$\xi$")
    axs[1, 1].set_xlabel("time [s]")
    axs[1, 1].set_ylabel("coordinate value")
    axs[1, 1].set_title(r"Path coordinates $(\eta,\xi)$")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    # --------------------------------------------------------
    # 5. Distance to path
    # --------------------------------------------------------

    axs[2, 0].plot(t, e_norm)
    axs[2, 0].set_xlabel("time [s]")
    axs[2, 0].set_ylabel(r"$\|\log_{\gamma(\eta)}(q)\|$")
    axs[2, 0].set_title("Distance to path")
    axs[2, 0].grid(True)

    # --------------------------------------------------------
    # 6. Control torques
    # --------------------------------------------------------

    axs[2, 1].plot(t, tau_hist[:, 0], label=r"$\tau_1$")
    axs[2, 1].plot(t, tau_hist[:, 1], label=r"$\tau_2$")
    axs[2, 1].set_xlabel("time [s]")
    axs[2, 1].set_ylabel("torque")
    axs[2, 1].set_title("Control torques")
    axs[2, 1].grid(True)
    axs[2, 1].legend()

    fig.suptitle("Two-Link Manipulator Path Following Summary", fontsize=16)
    fig.tight_layout()

    plt.show()
