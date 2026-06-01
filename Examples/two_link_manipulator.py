import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Manifolds.joints import Revolute
from Manifolds.manifold import RiemannianManifold
from Manifolds.product_manifold import ProductManifold
from path import CubicSplinePath, coordinate_circle_path



# Q = S^1 x S^1
CONFIG_MANIFOLD = ProductManifold(Revolute(), Revolute())

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
    Return M(q), the kinetic-energy metric for a standard planar 2R arm.

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
    """Return C(q, qdot) qdot."""
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
    """Return G(q)."""
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
    Solve M(q) qddot + C(q,qdot)qdot + G(q) = tau for qddot.
    """
    return np.linalg.solve(mass_matrix(q), tau - coriolis_vector(q, qdot) - gravity_vector(q))

def inverse_dynamics(q, qdot, qddot_cmd):
    """
    Computed-torque map from desired acceleration to torque.
    """
    return mass_matrix(q) @ qddot_cmd + (coriolis_vector(q, qdot) + gravity_vector(q))

def state_dot(state, tau):
    q = state[:, 0]
    qdot = state[:, 1]
    qddot = forward_dynamics(q, qdot, tau)
    return np.column_stack((qdot, qddot))

def wrap_state(state):
    """
    Wrap only the joint angles. This is topological projection, not the
    Riemannian exponential map.
    """
    q = CONFIG_MANIFOLD.project(state[:, 0])
    qdot = state[:, 1]
    return np.column_stack((q, qdot))


# Metrics used in the comparison

# Exact method:
# Q_EXACT.Log(p, q) solves the geodesic boundary-value problem for M(q).
Q_EXACT = RiemannianManifold(CONFIG_MANIFOLD, mass_matrix)

# approximate method:
# Q_APPROXIMATE.Log(p, q) uses the wrapped joint-space chart displacement, while
# Q_APPROXIMATE.inner/norm still use the kinetic-energy metric M(q).
Q_APPROXIMATE = RiemannianManifold(CONFIG_MANIFOLD, mass_matrix, use_analytic=True)

# Setup test paths

PATH_KIND = os.environ.get("PATH_KIND", "circle")


def make_spline_test_path(Q):
    waypoints = [
        np.array([0.0, 0.75]),
        np.array([0.75, 0.0]),
        np.array([0.0, -0.75]),
        np.array([-0.75, 0.0]),
    ]
    return CubicSplinePath(Q, waypoints, closed=True, bc_type="periodic")


def make_configuration_path(kind=PATH_KIND, Q=Q_APPROXIMATE):
    if kind == "circle":
        return coordinate_circle_path(Q, radius=0.8)
    if kind == "spline":
        return make_spline_test_path(Q)
    raise ValueError(f"Unknown path kind {kind!r}. Use 'circle' or 'spline'.")


configuration_path = make_configuration_path(PATH_KIND, Q_APPROXIMATE)


# Set up frame
def gamma(s):
    return configuration_path.eval(s)

def gamma_prime(s):
    return configuration_path.derivative(s)


# ============================================================
# eta-xi coordinate transforms
# ============================================================

def unit_tangent(Q, path, eta):
    """
    Returns normalized tangent vector
    """
    p = path.eval(eta)
    tangent = path.derivative(eta)
    return tangent / Q.norm(p, tangent)


def unit_normal(Q, p, tangent):
    """
    For 2D, the normal vecotr N satisfies T^T @ G @ N = 0

    Set alpha = G @ T, then alpha^T @ N = 0, 
    so the N is perpendicular to alpha in the Euclidean norm 
    """
    alpha = Q.metric(p) @ tangent
    normal = np.array([-alpha[1], alpha[0]])
    return normal / Q.norm(p, normal)


def normalize_eta(s, closed=True):
    if closed:
        return float(s % 1.0)
    return float(np.clip(s, 0.0, 1.0))


def minimize_eta(objective, eta_guess=None, window=0.1, global_grid=80, candidates=4):
    """
    One-dimensional closest-point search over eta.

    If eta_guess is provided, search locally. If not,
    do a coarse global scan first.
    """
    intervals = []

    if eta_guess is not None:
        # if we are provided with an eta_guess then we make a window about it and search locally
        intervals.append((eta_guess - window, eta_guess + window))
    else:
        grid = np.linspace(0.0, 1.0, global_grid, endpoint=False)

        # typically obkective(s) = d(gamma(s), q)^2
        values = np.array([objective(s) for s in grid])

        # the indicies of the n smallest objective vals 
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
            lambda s: objective(normalize_eta(s)),
            bounds=(a, b),
            method="bounded",
        )
        if result.fun < best_value:
            best_eta = normalize_eta(result.x)
            best_value = result.fun

    return best_eta

# Map psi: q -> (eta, xi)
def closest_eta_xi_exact(q, path, eta_guess=None):
    """
    Exact version:

        eta = argmin_s || Log_gamma(s)(q) ||_M^2
        xi  = Log_gamma(eta)(q)
    """
    q = CONFIG_MANIFOLD.project(q)

    def objective(s):
        p = path.eval(s)
        xi = Q_EXACT.Log(p, q)
        return Q_EXACT.squared_norm(p, xi)

    eta = minimize_eta(objective, eta_guess=eta_guess)
    p = path.eval(eta)
    xi = Q_EXACT.Log(p, q)

    return eta, p, xi

def closest_eta_xi_approximate(q, path, eta_guess=None):
    """
    Approximate local version:

        delta = wrapped joint-space displacement from gamma(s) to q
        eta   = argmin_s delta^T M(gamma(s)) delta
        xi    = delta
    """
    q = CONFIG_MANIFOLD.project(q)

    # TODO: check if this concides with lie group exponential/log...
    def objective(s):
        p = path.eval(s)
        xi = CONFIG_MANIFOLD.log(p, q)
        return Q_APPROXIMATE.squared_norm(p, xi)

    eta = minimize_eta(objective, eta_guess=eta_guess)
    p = path.eval(eta)
    xi = CONFIG_MANIFOLD.log(p, q)

    return eta, p, xi


# ============================================================
# Path-following controllers
# ============================================================

# Gains
controller_params = {
    "path_speed": 0.25,
    "eta_velocity_gain": 8.0,
    "xi_position_gain": 16.0,
    "xi_velocity_gain": 8.0,
}

# setup coordinates and frame
def closest_path_coordinates(q, path, eta_guess, method):
    if method == "exact":
        Q = Q_EXACT
        eta, p, xi_vec = closest_eta_xi_exact(q, path, eta_guess)
    elif method == "approximate":
        Q = Q_APPROXIMATE
        eta, p, xi_vec = closest_eta_xi_approximate(q, path, eta_guess)
    else:
        raise ValueError("method must be 'exact', of 'approximate'")

    tangent = unit_tangent(Q, path, eta)
    normal = unit_normal(Q, p, tangent)
    xi_scalar = Q.inner(p, xi_vec, normal)

    return Q, eta, p, tangent, normal, xi_scalar

# Phi: (eta, xi) -> q
def phi(path, Q, eta, xi, method):
    """
    Map (eta, xi) -> q by:
        q = Phi(eta, xi) = Exp_gamma(eta)(xi N(eta)).
    """
    p = path.eval(eta)
    tangent = unit_tangent(Q, path, eta)
    normal = unit_normal(Q, p, tangent)
    displacement = float(xi) * normal

    if method == "exact":
        return Q.Exp(p, displacement)
    if method == "approximate":
        return CONFIG_MANIFOLD.exp(p, displacement)
    raise ValueError("method must be 'exact', of 'approximate'")

# numerically compute D Phi
def phi_jacobian(path, Q, eta, xi, method, eps_eta=1e-5, eps_xi=1e-5):
    """
    Numerically compute D Phi(eta, xi).
    """
    q0 = phi(path, Q, eta, xi, method)

    q_eta_plus = phi(path, Q, eta + eps_eta, xi, method)
    q_eta_minus = phi(path, Q, eta - eps_eta, xi, method)
    d_eta = (
        CONFIG_MANIFOLD.log(q0, q_eta_plus)
        - CONFIG_MANIFOLD.log(q0, q_eta_minus)
    ) / (2.0 * eps_eta)

    q_xi_plus = phi(path, Q, eta, xi + eps_xi, method)
    q_xi_minus = phi(path, Q, eta, xi - eps_xi, method)
    d_xi = (
        CONFIG_MANIFOLD.log(q0, q_xi_plus)
        - CONFIG_MANIFOLD.log(q0, q_xi_minus)
    ) / (2.0 * eps_xi)

    return np.column_stack((d_eta, d_xi))

# numerically compute d/dt D phi @ ydot
def jacobian_dot_times_output_velocity(path, Q, eta, xi, ydot, method):
    """
    Directional derivative (d/dt D Phi) ydot.
    """
    if np.linalg.norm(ydot) < 1e-12:
        return np.zeros(Q.dim)

    eps_time = 1e-4
    eta_plus = eta + eps_time * ydot[0]
    xi_plus = xi + eps_time * ydot[1]
    eta_minus = eta - eps_time * ydot[0]
    xi_minus = xi - eps_time * ydot[1]

    J_plus = phi_jacobian(path, Q, eta_plus, xi_plus, method)
    J_minus = phi_jacobian(path, Q, eta_minus, xi_minus, method)

    Jdot = (J_plus - J_minus) / (2.0 * eps_time)
    return Jdot @ ydot

# eta isnt length parameterized so we need to convert rates
def desired_eta_rate(Q, path, eta):
    """
    Convert the desired metric speed along the path to eta_dot.
    """
    p = path.eval(eta)
    gamma_eta = path.derivative(eta)
    path_parameter_speed = Q.norm(p, gamma_eta)
    return controller_params["path_speed"] / max(path_parameter_speed, 1e-9)


def desired_eta_acceleration(Q, path, eta):
    """
    Calculate ddot eta
    """
    h = 1e-5
    rate_plus = desired_eta_rate(Q, path, eta + h)
    rate_minus = desired_eta_rate(Q, path, eta - h)
    d_rate_d_eta = (rate_plus - rate_minus) / (2.0 * h)
    rate = desired_eta_rate(Q, path, eta)
    return d_rate_d_eta * rate

####################################
# Controller
####################################
def computed_torque_path_controller(state, path, eta_guess, method):
    """
    Transverse feedback linearization in local path coordinates.

    Let y = [eta, xi] and q = Phi(y). The code computes

        qdot  = D Phi(y) ydot
        qddot = D Phi(y) yddot + d/dt(D Phi(y)) ydot

    and chooses yddot so eta tracks a path-speed reference while xi is
    stabilized to zero.
    """
    q = CONFIG_MANIFOLD.project(state[:, 0])
    qdot = state[:, 1]

    Q, eta, _, _, _, xi = closest_path_coordinates(
        q,
        path,
        eta_guess,
        method,
    )

    J = phi_jacobian(path, Q, eta, xi, method)
    eta_dot, xi_dot = np.linalg.solve(J, qdot)

    eta_dot_ref = desired_eta_rate(Q, path, eta)
    eta_ddot_ref = desired_eta_acceleration(Q, path, eta)
    eta_ddot_cmd = eta_ddot_ref + controller_params["eta_velocity_gain"] * (
        eta_dot_ref - eta_dot
    )

    xi_ddot_cmd = (
        -controller_params["xi_position_gain"] * xi
        -controller_params["xi_velocity_gain"] * xi_dot
    )

    ydot = np.array([eta_dot, xi_dot])
    yddot_cmd = np.array([eta_ddot_cmd, xi_ddot_cmd])

    qddot_feedforward = jacobian_dot_times_output_velocity(
        path,
        Q,
        eta,
        xi,
        ydot,
        method,
    )
    qddot_cmd = J @ yddot_cmd + qddot_feedforward
    tau = inverse_dynamics(q, qdot, qddot_cmd)

    return tau, eta, xi

# Integrator steps
def euler_step(state, tau, dt):
    next_state = state + dt * state_dot(state, tau)
    return wrap_state(next_state)

def rk4_step(state, tau_func, dt):
    def f(x):
        tau = tau_func(x)
        return state_dot(x, tau)

    k1 = f(state)
    k2 = f(wrap_state(state + 0.5 * dt * k1))
    k3 = f(wrap_state(state + 0.5 * dt * k2))
    k4 = f(wrap_state(state + dt * k3))

    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return wrap_state(next_state)

####################################
# Main sim loop
####################################
def simulate_path_following(
    state0,
    dt=0.005,
    T_final=0.05,
    use_rk4=False,
    method="approximate",
    path=configuration_path,
):
    """
    Simulate closed-loop path following and record wall-clock time per step.
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
        "step_time": [],
    }

    for k in range(n_steps):
        print("step " + str(k) + " of " + str(n_steps))
        step_start = time.perf_counter()
        t = k * dt

        tau, eta, xi = computed_torque_path_controller(
            state,
            path,
            eta_guess,
            method=method,
        )
        eta_guess = eta

        history["t"].append(t)
        history["q"].append(state[:, 0].copy())
        history["qdot"].append(state[:, 1].copy())
        history["eta"].append(eta)
        history["xi"].append(xi)
        history["tau"].append(tau.copy())

        if use_rk4:
            def tau_func(x):
                tau_x, _, _ = computed_torque_path_controller(
                    x,
                    path,
                    eta_guess,
                    method=method,
                )
                return tau_x

            state = rk4_step(state, tau_func, dt)
        else:
            state = euler_step(state, tau, dt)

        history["step_time"].append(time.perf_counter() - step_start)

    for key in history:
        history[key] = np.array(history[key])

    return history

# Plottint
def timing_summary(name, history):
    step_time = history["step_time"]
    if len(step_time) == 0:
        print(f"{name}: no steps")
        return

    print(f"\n{name}")
    print(f"  steps:          {len(step_time)}")
    print(f"  total walltime: {np.sum(step_time):.6f} s")
    print(f"  first step:     {step_time[0]:.6f} s")
    print(f"  mean step:      {np.mean(step_time):.6f} s")
    print(f"  median step:    {np.median(step_time):.6f} s")
    print(f"  min step:       {np.min(step_time):.6f} s")
    print(f"  max step:       {np.max(step_time):.6f} s")
    if len(step_time) > 1:
        print(f"  mean after 1st: {np.mean(step_time[1:]):.6f} s")

def plot_single_summary(history, path, title):
    import matplotlib.pyplot as plt

    t = history["t"]
    q_hist = history["q"]
    qdot_hist = history["qdot"]
    eta_hist = history["eta"]
    xi_hist = history["xi"]
    tau_hist = history["tau"]
    step_time = history["step_time"]

    path_grid = np.linspace(0.0, 1.0, 400)
    path_points = np.array([path.eval(s) for s in path_grid])

    fig, axs = plt.subplots(3, 2, figsize=(14, 12))

    axs[0, 0].plot(path_points[:, 0], path_points[:, 1], label="path")
    axs[0, 0].plot(q_hist[:, 0], q_hist[:, 1], label="q(t)")
    axs[0, 0].scatter(q_hist[0, 0], q_hist[0, 1], marker="o", label="start")
    axs[0, 0].scatter(q_hist[-1, 0], q_hist[-1, 1], marker="x", label="end")
    axs[0, 0].axis("equal")
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    axs[0, 0].set_title("Configuration-space path")

    axs[0, 1].plot(t, q_hist[:, 0], label="theta1")
    axs[0, 1].plot(t, q_hist[:, 1], label="theta2")
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    axs[0, 1].set_title("Joint angles")

    axs[1, 0].plot(t, qdot_hist[:, 0], label="theta1_dot")
    axs[1, 0].plot(t, qdot_hist[:, 1], label="theta2_dot")
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    axs[1, 0].set_title("Joint velocities")

    axs[1, 1].plot(t, eta_hist, label="eta")
    axs[1, 1].plot(t, xi_hist, label="xi")
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    axs[1, 1].set_title("Path coordinates")

    axs[2, 0].plot(t, step_time)
    axs[2, 0].grid(True)
    axs[2, 0].set_title("Wall-clock time per step")
    axs[2, 0].set_xlabel("simulation time [s]")
    axs[2, 0].set_ylabel("wall time [s]")

    axs[2, 1].plot(t, tau_hist[:, 0], label="tau1")
    axs[2, 1].plot(t, tau_hist[:, 1], label="tau2")
    axs[2, 1].grid(True)
    axs[2, 1].legend()
    axs[2, 1].set_title("Control torques")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    plt.show()

def plot_comparison(exact_hist, approximate_hist, path):
    import matplotlib.pyplot as plt

    path_grid = np.linspace(0.0, 1.0, 400)
    path_points = np.array([path.eval(s) for s in path_grid])

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(path_points[:, 0], path_points[:, 1], label="path")
    axs[0, 0].plot(exact_hist["q"][:, 0], exact_hist["q"][:, 1], label="exact")
    axs[0, 0].plot(approximate_hist["q"][:, 0], approximate_hist["q"][:, 1], label="approximate")
    axs[0, 0].axis("equal")
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    axs[0, 0].set_title("Configuration-space trajectory")

    axs[0, 1].plot(exact_hist["t"], exact_hist["xi"], label="exact")
    axs[0, 1].plot(approximate_hist["t"], approximate_hist["xi"], label="approximate")
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    axs[0, 1].set_title("Transverse coordinate xi")

    axs[1, 0].plot(exact_hist["t"], exact_hist["step_time"], label="exact")
    axs[1, 0].plot(approximate_hist["t"], approximate_hist["step_time"], label="approximate")
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    axs[1, 0].set_title("Wall-clock time per simulation step")

    axs[1, 1].plot(exact_hist["t"], exact_hist["eta"], label="exact")
    axs[1, 1].plot(approximate_hist["t"], approximate_hist["eta"], label="approximate")
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    axs[1, 1].set_title("Path coordinate eta")

    fig.tight_layout()
    plt.show()

def plot_two_method_comparison(first_hist, second_hist, path, first_label, second_label):
    import matplotlib.pyplot as plt

    path_grid = np.linspace(0.0, 1.0, 400)
    path_points = np.array([path.eval(s) for s in path_grid])

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(path_points[:, 0], path_points[:, 1], label="path")
    axs[0, 0].plot(first_hist["q"][:, 0], first_hist["q"][:, 1], label=first_label)
    axs[0, 0].plot(second_hist["q"][:, 0], second_hist["q"][:, 1], label=second_label)
    axs[0, 0].axis("equal")
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    axs[0, 0].set_title("Configuration-space trajectory")

    axs[0, 1].plot(first_hist["t"], first_hist["xi"], label=first_label)
    axs[0, 1].plot(second_hist["t"], second_hist["xi"], label=second_label)
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    axs[0, 1].set_title("Transverse coordinate xi")

    axs[1, 0].plot(first_hist["t"], first_hist["step_time"], label=first_label)
    axs[1, 0].plot(second_hist["t"], second_hist["step_time"], label=second_label)
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    axs[1, 0].set_title("Wall-clock time per simulation step")

    axs[1, 1].plot(first_hist["t"], first_hist["eta"], label=first_label)
    axs[1, 1].plot(second_hist["t"], second_hist["eta"], label=second_label)
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    axs[1, 1].set_title("Path coordinate eta")

    fig.tight_layout()
    plt.show()

def compare_exact_and_approximate(
    state0,
    dt=0.005,
    T_final=0.025,
    use_rk4=False,
    path=configuration_path,
):
    """
    Run the exact geodesic controller and approximate local controller on the same
    two-link arm and print per-step timing statistics.
    """
    print("Running exact geodesic-BVP metric simulation...")
    exact = simulate_path_following(
        state0,
        dt=dt,
        T_final=T_final,
        use_rk4=use_rk4,
        method="exact",
        path=path,
    )

    print("Running approximate local metric simulation...")
    approximate = simulate_path_following(
        state0,
        dt=dt,
        T_final=T_final,
        use_rk4=use_rk4,
        method="approximate",
        path=path,
    )

    timing_summary("Exact geodesic BVP metric", exact)
    timing_summary("Approximate local metric", approximate)

    if len(exact["step_time"]) and len(approximate["step_time"]):
        ratio = np.mean(exact["step_time"]) / np.mean(approximate["step_time"])
        print(f"\nMean-step speedup: {ratio:.1f}x")

    return exact, approximate


# Example

if __name__ == "__main__":
    q0 = np.array([0.1, 0.1])
    qdot0 = np.array([0.0, 0.0])
    state0 = np.column_stack((q0, qdot0))

    dt = float(os.environ.get("DT", "0.005"))
    T_final = float(os.environ.get("T_FINAL", "0.025"))
    use_rk4 = os.environ.get("USE_RK4", "0") == "1"
    sim_mode = os.environ.get("SIM_MODE", "compare")
    show_plot = os.environ.get("SHOW_PLOT", "0") == "1"

    print(f"PATH_KIND = {PATH_KIND}")
    print(f"SIM_MODE = {sim_mode}")
    print(f"dt = {dt}")
    print(f"T_final = {T_final}")
    print(f"use_rk4 = {use_rk4}")

    if sim_mode == "compare":
        exact_hist, approximate_hist = compare_exact_and_approximate(
            state0,
            dt=dt,
            T_final=T_final,
            use_rk4=use_rk4,
            path=configuration_path,
        )
        if show_plot:
            plot_comparison(exact_hist, approximate_hist, configuration_path)
        if show_plot:
            plot_two_method_comparison(
                approximate_hist,
                configuration_path,
                "approximate",
            )
    elif sim_mode in ("approximate", "exact"):
        print(f"Running {sim_mode} simulation...")
        hist = simulate_path_following(
            state0,
            dt=dt,
            T_final=T_final,
            use_rk4=use_rk4,
            method=sim_mode,
            path=configuration_path,
        )
        timing_summary(f"{sim_mode} simulation", hist)
        if show_plot:
            plot_single_summary(hist, configuration_path, f"{sim_mode} simulation")
    else:
        raise ValueError("SIM_MODE must be 'compare', 'approximate', 'exact'.")
