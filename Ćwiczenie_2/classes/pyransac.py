from __future__ import annotations

import numpy as np
from pyransac3d import Plane

_SQRT2_INV = 1 / np.sqrt(2)


def _orientation(normal: np.ndarray) -> str:
    return "horizontal" if abs(normal[2]) / np.linalg.norm(normal) >= _SQRT2_INV else "vertical"


class PlaneRANSAC:

    def __init__(self, num_iterations: int = 100, threshold: float = 0.01):
        self.num_iterations = num_iterations
        self.threshold = threshold
        self.coefficients: tuple[float, float, float, float] | None = None
        self.inliers: np.ndarray | None = None

    def fit(self, points: np.ndarray) -> None:
        coeffs, idx = Plane().fit(points, thresh=self.threshold, maxIteration=self.num_iterations)
        self.coefficients = tuple(float(c) for c in coeffs)
        self.inliers = points[idx]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _distances_from_plane(coeffs: tuple[float, float, float, float], points: np.ndarray) -> np.ndarray:
        a, b, c, d = coeffs
        return np.abs(points @ np.array([a, b, c]) + d) / np.sqrt(a * a + b * b + c * c)

    def distances(self, points: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self._distances_from_plane(self.coefficients, points)

    # ------------------------------------------------------------------
    # Pretty printing (DBSCAN-style)
    # ------------------------------------------------------------------
    def print_result(self, cluster: np.ndarray, idx: int, planar_tol: float) -> None:
        if self.coefficients is None:
            raise RuntimeError("Call .fit() before printing results.")

        mean_dist = float(self.distances(cluster).mean())
        if mean_dist > planar_tol:
            print(f"Cloud {idx} is not a plane.")
            return

        normal = np.asarray(self.coefficients[:3])
        print(f"Cloud {idx} is a plane")
        print(f"The plane is {_orientation(normal)}.")
