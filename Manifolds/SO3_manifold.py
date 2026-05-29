import numpy as np
from scipy.spatial.transform import Rotation

from Manifolds.manifold import Manifold


def _skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


class SO3(Manifold):
    """SO(3) represented by rotation-vector coordinates."""

    def __init__(self):
        self.dim = 3

    def coordinate(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape == (3, 3):
            return Rotation.from_matrix(q).as_rotvec()
        if q.shape == (3,):
            return q
        raise ValueError(f"Expected rotation matrix or rotvec, got shape {q.shape}.")

    def project(self, q):
        return Rotation.from_rotvec(self.coordinate(q)).as_rotvec()

    def as_matrix(self, q):
        return Rotation.from_rotvec(self.project(q)).as_matrix()

    def exp(self, q, v):
        r = Rotation.from_rotvec(self.project(q))
        return (r * Rotation.from_rotvec(np.asarray(v, dtype=float).reshape(3))).as_rotvec()

    def log(self, q, p):
        rq = Rotation.from_rotvec(self.project(q))
        rp = Rotation.from_rotvec(self.project(p))
        return (rq.inv() * rp).as_rotvec()

    def dist(self, q, p):
        return float(np.linalg.norm(self.log(q, p)))

    def L(self, q, p):
        return (Rotation.from_rotvec(self.project(q)) * Rotation.from_rotvec(self.project(p))).as_rotvec()

    def R(self, q, p):
        return (Rotation.from_rotvec(self.project(p)) * Rotation.from_rotvec(self.project(q))).as_rotvec()

    def TL(self, q, v, b):
        return self.as_matrix(q) @ _skew(v) @ self.as_matrix(b)

    def TR(self, q, v, b):
        return self.as_matrix(b) @ _skew(v) @ self.as_matrix(q)
