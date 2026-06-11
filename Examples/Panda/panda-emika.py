import os
import subprocess
import mujoco
import mujoco.viewer
import time
from pathlib import Path
import numpy as np

from Numerics.closest_point import minimize_eta
from Manifolds.joints import Revolute
from Manifolds.joints import wrap_angle
from Manifolds.manifold import RiemannianManifold
from Manifolds.product_manifold import ProductManifold

# Initial config of 7 joints
q_home = np.array([
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
])

# Controller
controller_params = {
    "path_speed": 0.15,
    "eta_velocity_gain": 6.0,
    "xi_position_gain": 20.0,
    "xi_velocity_gain": 10.0,
    "tau_limit": np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]),
}

# run: PYTHONPATH=. T_FINAL=10 SHOW_PLOT=1 mjpython Examples/Panda/panda-emika.py
# 
MODEL_PATH = Path.home() / "Desktop" / "Robotics" / "mujoco" / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml"
DEFAULT_PLOT_PATH = Path(__file__).with_name("panda_plots.png")

# Dynamics ##############################
# M(q)q'' + C(q, q')q' + grad P(q) = tau

# M(q)
def mass_matrix(model, data):
    # By deafult mujoco stores the inertia in data.qM as a 'sparse matrix',
    # we want it to be dense for calculations.
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM) # shape = (9,9)
    # print("dim(mj_fullM) = ", np.shape(M))
    return M[:7, :7]

# KE helpers
def inner(M, u, v):
    return float(u.T @ M @ v)

def norm(M, u):
    return np.sqrt(max(inner(M, u, u), 0.0))

# Evaluates metric at given point
def metric_at_base(model, data, p):
    q_old = data.qpos.copy()
    dq_old = data.qvel.copy()

    # TODO: check if this is correct. Shouldnt M(q, q') = M(q, 0)?
    data.qpos[:7] = p
    data.qvel[:7] = 0.0
    mujoco.mj_forward(model, data)

    M = mass_matrix(model, data)

    data.qpos[:] = q_old
    data.qvel[:] = dq_old

    mujoco.mj_forward(model, data)

    return M

# C(q, q')q' + grad P(q)
def drift_forces(data):
    return data.qfrc_bias[:7].copy()

# qddot -> tau
def inverse_dynamics(model, data, qddot_cmd):
    M = mass_matrix(model, data)
    d = drift_forces(data)
    return M @ qddot_cmd + d
########################################

config = ProductManifold(Revolute(), Revolute(), 
                                  Revolute(), Revolute(), 
                                  Revolute(), Revolute(), 
                                  Revolute())

# TODO: make these work with the exp and log functions in the ProductManifold class
def flat_log(p, q):
    return wrap_angle(q - p)
def flat_exp(p, v):
    return wrap_angle(p + v)

# Path ##############################
q_center = np.array([
    0.0,
   -0.75,
    0.0,
   -2.5,
    0.0,
    1.5,
    0.75,
])
a1 = 1.0
a2 = 1.5


# TODO: We should have a better way of generating paths... Make better interpolator, real time curve reframing??
# Path is ellipse in q1-q2 plane with center q_center.
def gamma(eta):
    theta = 2 * np.pi * eta
    q = q_center.copy()

    q[1] += a1 * np.sin(theta)
    q[2] += a2 * np.cos(theta)
    return wrap_angle(q)

def gamma_prime(eta):
    theta = 2 * np.pi * eta
    dq = np.zeros(7)

    dq[0] += a1 * 2.0 * np.pi * np.cos(theta)
    dq[1] += -a2 * 2.0 * np.pi * np.sin(theta)
    return dq

# TODO: this should be done with the method outlined in the SO(3) paper
def frame(model, data, eta):
    p = gamma(eta)
    M = metric_at_base(model, data, p)

    T = gamma_prime(eta)
    T = T / max(norm(M, T), 1e-9) # normalize
    basis = [T]

    # Start from Euclidean coordinate basis and Gram-Schmidt with M inner product.
    for i in range(7):
        v = np.zeros(7)
        v[i] = 1.0
        for e in basis:
            v = v - inner(M, v, e) * e
        nrm = norm(M, v)
        if nrm > 1e-8:
            basis.append(v / nrm)
        if len(basis) == 7:
            break
    E = np.column_stack(basis)
    return p, E
########################################


def closest_path_coordinates(model, data, q, eta_guess=None):
    q = wrap_angle(q)

    # objective = M_gamma(s)(q - gamma(s), q - gamma(s))
    def objective(s):
        p = gamma(s)
        M = metric_at_base(model, data, p)
        xi = flat_log(p, q)
        return inner(M, xi, xi)
    
    eta = minimize_eta(objective, eta_guess=eta_guess)

    p, E = frame(model, data, eta)
    M = metric_at_base(model, data, p)
    xi_full = flat_log(p, q)

    # Coordinates in M-orthonormal frame.
    coords = E.T @ M @ xi_full
    _ = coords[0] # we take eta to be the distance minimizing point so ideally this is ~0
    xi_normal = coords[1:]

    return eta, xi_normal

# phi: (eta, xi) -> q
def phi(model, data, eta, xi_normal):
    p, E = frame(model, data, eta)
    displacement = E[:, 1:] @ xi_normal

    return flat_exp(p, displacement)

# Approximate Jacobian by finite differences
def phi_jacobian(model, data, eta, xi_normal, eps_eta=1e-5, eps_xi=1e-5):
    q0 = phi(model, data, eta, xi_normal)
    J = np.zeros((7, 7))

    q_plus = phi(model, data, eta + eps_eta, xi_normal)
    q_minus = phi(model, data, eta - eps_eta, xi_normal)

    J[:, 0] = (flat_log(q0, q_plus) - flat_log(q0, q_minus)) / (2.0 * eps_eta)

    for i in range(6):
        xi_plus = xi_normal.copy()
        xi_minus = xi_normal.copy()

        xi_plus[i] += eps_xi
        xi_minus[i] -= eps_xi

        q_plus = phi(model, data, eta, xi_plus)
        q_minus = phi(model, data, eta, xi_minus)

        J[:, i + 1] = (flat_log(q0, q_plus) - flat_log(q0, q_minus)) / (2.0 * eps_xi)

    return J

def jacobian_dot_times_ydot(model, data, eta, xi_normal, ydot):
    if np.linalg.norm(ydot) < 1e-12:
        return np.zeros(7)
    eps_time = 1e-4

    eta_plus = eta + eps_time * ydot[0]
    eta_minus = eta - eps_time * ydot[0]

    xi_plus = xi_normal + eps_time * ydot[1:]
    xi_minus = xi_normal - eps_time * ydot[1:]

    J_plus = phi_jacobian(model, data, eta_plus, xi_plus)
    J_minus = phi_jacobian(model, data, eta_minus, xi_minus)

    Jdot = (J_plus - J_minus) / (2.0 * eps_time)

    return Jdot @ ydot

def desired_eta_rate(model, data, eta):
    M = metric_at_base(model, data, gamma(eta))
    gp = gamma_prime(eta)
    path_parameter_speed = norm(M, gp)

    return controller_params["path_speed"] / max(path_parameter_speed, 1e-9)

def desired_eta_acceleration(model, data, eta):
    h = 1e-5

    rate_plus = desired_eta_rate(model, data, eta + h)
    rate_minus = desired_eta_rate(model, data, eta - h)

    d_rate_d_eta = (rate_plus - rate_minus) / (2.0 * h)

    rate = desired_eta_rate(model, data, eta)

    return d_rate_d_eta * rate

def computed_torque_path_controller(model, data, eta_guess):
    q = wrap_angle(data.qpos[:7].copy())
    dq = data.qvel[:7].copy()

    eta, xi = closest_path_coordinates(
        model,
        data,
        q,
        eta_guess=eta_guess,
    )

    J = phi_jacobian(model, data, eta, xi)

    # ydot = [eta_dot, xi_dot]
    ydot = np.linalg.solve(J, dq)

    eta_dot = ydot[0]
    xi_dot = ydot[1:]

    eta_dot_ref = desired_eta_rate(model, data, eta)
    eta_ddot_ref = desired_eta_acceleration(model, data, eta)

    eta_ddot_cmd = eta_ddot_ref + controller_params["eta_velocity_gain"] * (
        eta_dot_ref - eta_dot
    )
    xi_ddot_cmd = (
        -controller_params["xi_position_gain"] * xi
        -controller_params["xi_velocity_gain"] * xi_dot
    )

    yddot_cmd = np.concatenate([
        np.array([eta_ddot_cmd]),
        xi_ddot_cmd,
    ])

    Jdot_ydot = jacobian_dot_times_ydot(model, data, eta, xi, ydot)
    qddot_cmd = J @ yddot_cmd + Jdot_ydot
    tau = inverse_dynamics(model, data, qddot_cmd)

    tau = np.clip(
        tau,
        -controller_params["tau_limit"],
        controller_params["tau_limit"],
    )

    return tau, eta, xi

# Logging 
def plot_eta_xi(history, save_path=None):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    t = np.array(history["t"])
    eta = np.array(history["eta"])
    xi = np.array(history["xi"])
    if len(t) == 0:
        print("No eta/xi data to plot.")
        return

    if xi.ndim == 1:
        xi = xi.reshape(-1, 1)

    xi_norm = np.linalg.norm(xi, axis=1)
    step_time = np.array(history["step_time"])

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axs[0].plot(t, eta)
    axs[0].grid(True)
    axs[0].set_ylabel("eta")
    axs[0].set_title("Path coordinate eta")

    axs[1].plot(t, xi_norm, label="||xi||")
    axs[1].grid(True)
    axs[1].set_ylabel("||xi||")
    axs[1].set_title("Transverse error norm")

    for i in range(xi.shape[1]):
        axs[2].plot(t, xi[:, i], label=f"xi_{i+1}")

    axs[2].grid(True)
    axs[2].legend(loc="upper right", ncol=3)
    axs[2].set_xlabel("time [s]")
    axs[2].set_ylabel("xi_i")
    axs[2].set_title("Transverse coordinates")

    fig.tight_layout()

    save_path = DEFAULT_PLOT_PATH if save_path is None else Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved eta/xi plot to {save_path}")

    
    subprocess.run(["open", str(save_path)], check=False)

    plt.close(fig)

def timing_summary(history):
    step_time = np.array(history["step_time"])

    if len(step_time) == 0:
        print("No data recorded.")
        return

    print("\nTiming summary")
    print(f"  steps:       {len(step_time)}")
    print(f"  total:       {np.sum(step_time):.6f} s")
    print(f"  mean step:   {np.mean(step_time):.6f} s")
    print(f"  median step: {np.median(step_time):.6f} s")
    print(f"  max step:    {np.max(step_time):.6f} s")
####################



# Main MuJoCo sim
def main():
    t_final = os.environ.get("T_FINAL")
    t_final = None if t_final is None else float(t_final)
    plot_path = os.environ.get("PLOT_PATH")
    plot_path = DEFAULT_PLOT_PATH if plot_path is None else Path(plot_path)

    # Load model
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # The chosen model has built in joint pd control, to supply my own torques set the joint PD gains to zero.
    model.actuator_gainprm[:7, :] = 0.0
    model.actuator_biasprm[:7, :] = 0.0

    print("Loaded Panda model")
    print("number of generalize coordinates (nq) =", model.nq)
    print("numer of degrees of freedom (nv) =", model.nv)
    print("number of actuators (nq) =", model.nu)

    # Initial conditions
    data.qpos[:7] = q_center + np.array([0.1, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0])
    data.qvel[:7] = 0.0

    # Keep gripper closed.
    if model.nq >= 9:
        data.qpos[7:9] = 0.0

    mujoco.mj_forward(model, data)

    eta_guess = None

    history = {
        "t": [],
        "eta": [],
        "xi": [],
        "tau": [],
        "q": [],
        "qdot": [],
        "step_time": [],
    }

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                if t_final is not None and data.time >= t_final:
                    break

                step_start = time.perf_counter()
                tau, eta, xi = computed_torque_path_controller(
                    model,
                    data,
                    eta_guess,
                )

                eta_guess = eta

                data.qfrc_applied[:] = 0.0
                data.qfrc_applied[:7] = tau

                # Gripper actuator; hold neutral.
                if model.nu >= 8:
                    data.ctrl[7] = 0.0

                mujoco.mj_step(model, data)
                viewer.sync()

                step_time = time.perf_counter() - step_start

                history["t"].append(data.time)
                history["eta"].append(eta)
                history["xi"].append(xi.copy())
                history["tau"].append(tau.copy())
                history["q"].append(data.qpos[:7].copy())
                history["qdot"].append(data.qvel[:7].copy())
                history["step_time"].append(step_time)

    except KeyboardInterrupt:
        print("\nInterrupted; plotting recorded eta/xi history.")
    finally:
        # Runs after the MuJoCo viewer closes or T_FINAL is reached.
        for key in history:
            history[key] = np.array(history[key])

        timing_summary(history)
        plot_eta_xi(history, save_path=plot_path)
    
if __name__ == "__main__":

    main()
