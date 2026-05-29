import unittest

import numpy as np

from Manifolds.joints import Revolute
from Manifolds.manifold import Euclidean, RiemannianManifold
from Manifolds.product_manifold import ProductManifold
from Numerics.closest_point import closest_point
from path import CubicSplinePath, ParametricPath, Path, coordinate_circle_path


class ManifoldTests(unittest.TestCase):
    def test_product_revolute_log_exp_round_trip(self):
        M = ProductManifold(Revolute(), Revolute())
        Q = RiemannianManifold(M, np.eye(2))

        q = np.array([3.0, -3.0])
        p = np.array([-3.0, 3.0])
        xi = Q.Log(q, p)

        np.testing.assert_allclose(Q.Exp(q, xi), M.project(p), atol=1e-12)
        self.assertLess(np.linalg.norm(xi), 0.5)

    def test_bvp_flat_metric_matches_straight_line_log(self):
        Q = RiemannianManifold(Euclidean(2), lambda q: np.eye(2), use_analytic=False)

        xi = Q.Log(np.array([0.0, 0.0]), np.array([1.0, 2.0]))

        np.testing.assert_allclose(xi, np.array([1.0, 2.0]), atol=1e-5)
        np.testing.assert_allclose(Q.Exp(np.zeros(2), xi), np.array([1.0, 2.0]), atol=1e-5)


class PathTests(unittest.TestCase):
    def test_piecewise_path_eval_uses_segment_arclength(self):
        Q = RiemannianManifold(Euclidean(2), np.eye(2))
        path = Path(
            Q,
            [
                np.array([0.0, 0.0]),
                np.array([1.0, 0.0]),
                np.array([1.0, 1.0]),
            ],
        )

        np.testing.assert_allclose(path.eval(0.0), np.array([0.0, 0.0]))
        np.testing.assert_allclose(path.eval(0.5), np.array([1.0, 0.0]))
        np.testing.assert_allclose(path.eval(1.0), np.array([1.0, 1.0]))

    def test_closest_point_on_line(self):
        Q = RiemannianManifold(Euclidean(2), np.eye(2))
        path = ParametricPath(
            Q,
            lambda s: np.array([s, 0.0]),
            lambda s: np.array([1.0, 0.0]),
            closed=False,
        )

        result = closest_point(Q, np.array([0.25, 0.5]), path)

        self.assertAlmostEqual(result.eta, 0.25, places=5)
        np.testing.assert_allclose(result.point, np.array([0.25, 0.0]), atol=1e-5)
        np.testing.assert_allclose(result.xi, np.array([0.0, 0.5]), atol=1e-5)
        self.assertAlmostEqual(result.normality_residual, 0.0, places=6)

    def test_cubic_spline_path_is_smooth_at_waypoint(self):
        Q = RiemannianManifold(Euclidean(2), np.eye(2))
        path = CubicSplinePath(
            Q,
            [
                np.array([0.0, 0.0]),
                np.array([0.5, 0.8]),
                np.array([1.0, 0.0]),
                np.array([1.5, -0.2]),
            ],
            bc_type="natural",
        )

        knot = path.knots[1]
        left = path.derivative(knot - 1e-6)
        right = path.derivative(knot + 1e-6)

        np.testing.assert_allclose(left, right, atol=1e-4)

    def test_fixed_circle_path_factory(self):
        Q = RiemannianManifold(Euclidean(2), np.eye(2))
        path = coordinate_circle_path(Q, radius=0.8)

        np.testing.assert_allclose(path.eval(0.0), path.eval(1.0), atol=1e-12)
        self.assertGreater(np.linalg.norm(path.derivative(0.0)), 1.0)


class TwoLinkExampleTests(unittest.TestCase):
    def test_controller_outputs_finite_torque_and_reduces_xi_short_term(self):
        from Examples.two_link_manipulator import path_following_controller, simulate_path_following

        q0 = np.array([0.2, 0.7])
        state0 = np.column_stack((q0, np.zeros(2)))

        tau, eta, xi = path_following_controller(state0)
        self.assertTrue(np.all(np.isfinite(tau)))
        self.assertTrue(np.isfinite(eta))
        self.assertTrue(np.isfinite(xi))

        hist = simulate_path_following(state0, dt=0.005, T_final=0.05, use_rk4=True)
        self.assertLess(abs(hist["xi"][-1]), abs(hist["xi"][0]))


if __name__ == "__main__":
    unittest.main()
