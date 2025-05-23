import numpy as np

class AnalyzeRANSAC:
    def __init__(self, num_iterations, threshold):
        self.num_iterations = num_iterations
        self.threshold = threshold

    def fit(self, data):
        best_eq = None
        best_inliers = []

        for _ in range(self.num_iterations):
            sample = data[np.random.choice(data.shape[0], 3, replace=False)]
            plane_eq = self._plane_equation(sample)

            distances = self._distances_from_plane(plane_eq, data)
            inliers = data[distances < self.threshold]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_eq = plane_eq

        self.inliers = best_inliers
        self.coefficients = best_eq

    def _plane_equation(self, points):
        p1, p2, p3 = points
        v1 = p3 - p1
        v2 = p2 - p1
        cross_product = np.cross(v1, v2)
        a, b, c = cross_product
        d = -(cross_product @ p1)
        return a, b, c, d

    def _distances_from_plane(self, plane_eq, points):
        a, b, c, d = plane_eq
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        return np.abs(a*x + b*y + c*z + d) / np.sqrt(a**2 + b**2 + c**2)