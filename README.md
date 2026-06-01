# Path-Following Control

Riemannian path-following control experiments for fully actuated robot models, with metric closest-point projection, eta/xi path coordinates, and computed-torque control examples.

## Usage

Run commands from the repository root:

```bash
cd /Users/jaiden/Desktop/Robotics/Path-Following-Contol
```

Install the Python dependencies if needed:

```bash
python3 -m pip install numpy scipy matplotlib
```

Run the test suite:

```bash
python3 -m unittest discover -v
```

Run the two-link manipulator timing comparison:

```bash
python3 Examples/two_link_manipulator.py
```

This compares two kinetic-energy-metric simulations:

- `exact`: closest-point projection uses geodesic BVP solves for `Log`.
- `fast`: closest-point projection uses wrapped joint displacement with the kinetic-energy metric inner product.
- `lie`: closest-point projection uses the Lie group logarithm/exponential on
  \(S^1 \times S^1\). For the two-link torus this is the same wrapped angle
  formula as `fast`, but it is named explicitly as a Lie group construction.

Both simulations use the same transverse feedback-linearizing controller in
local path coordinates. The fast simulation is the one to use for interactive
plots and tuning:

```bash
SIM_MODE=fast SHOW_PLOT=1 T_FINAL=5.0 python3 Examples/two_link_manipulator.py
```

Run only the fast simulation:

```bash
SIM_MODE=fast python3 Examples/two_link_manipulator.py
```

Run only the exact simulation:

```bash
SIM_MODE=exact python3 Examples/two_link_manipulator.py
```

Run only the Lie group simulation:

```bash
SIM_MODE=lie python3 Examples/two_link_manipulator.py
```

Compare the fast and Lie group methods:

```bash
SIM_MODE=compare_lie python3 Examples/two_link_manipulator.py
```

To try different non-self-intersecting test paths:

```bash
PATH_KIND=circle python3 Examples/two_link_manipulator.py
PATH_KIND=spline python3 Examples/two_link_manipulator.py
```

To change the timing horizon or use RK4:

```bash
T_FINAL=0.05 DT=0.005 python3 Examples/two_link_manipulator.py
USE_RK4=1 T_FINAL=0.01 python3 Examples/two_link_manipulator.py
```

To show plots:

```bash
SHOW_PLOT=1 python3 Examples/two_link_manipulator.py
SIM_MODE=fast SHOW_PLOT=1 python3 Examples/two_link_manipulator.py
```

The fast-only plot includes the configuration-space path, joint angles, joint
velocities, eta/xi coordinates, wall-clock step time, and control torques. The
exact simulation is mainly a timing/reference check; even one step can take
several seconds because it repeatedly solves geodesic boundary-value problems.

The math note comparing exact and fast coordinates is in
`docs/exact_vs_fast_path_following.tex`.

## Path Options

- `Path`: piecewise geodesic interpolation through waypoints.
- `CubicSplinePath`: smooth coordinate cubic spline through waypoints, projected back onto the manifold.
- `ParametricPath`: fixed or user-defined function path `gamma(s)`.
- Test factory: `coordinate_circle_path`.

## Notes

The current controller assumes a fully actuated second-order robot model:

```text
M(q) qddot + bias(q, qdot) = u
```

In `Examples/two_link_manipulator.py`, the controller builds the local chart
`q = Phi(eta, xi)`, computes `D Phi`, recovers `(eta_dot, xi_dot)`, chooses
second-order eta/xi dynamics, and maps the result back to `qddot_cmd` before
calling inverse dynamics. This is the part to read if you want the clearest
connection to the transverse feedback-linearization theory.

For curved or configuration-dependent metrics, the code can fall back to numerical geodesic solves. For standard joint spaces with constant metrics, analytic `Exp` and `Log` are used.
