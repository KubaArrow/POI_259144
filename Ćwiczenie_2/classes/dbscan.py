import csv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class AnalyzeDBSCAN:
    def __init__(self, file_path):
        self.file_path = file_path
        self.cloud_points = None

    def load_cloud_points(self):
        self.cloud_points = []
        with open(self.file_path, 'r') as file:
            reader = csv.reader(file, delimiter=' ')
            for row in reader:
                point = [float(row[0]), float(row[1]), float(row[2])]
                self.cloud_points.append(point)

    def separate_clusters(self, k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(self.cloud_points)
        labels = kmeans.labels_
        unique_labels = set(labels)
        clusters = []
        for label in unique_labels:
            cluster = [self.cloud_points[i] for i in range(len(self.cloud_points)) if labels[i] == label]
            clusters.append(cluster)
        return clusters

    def fit_plane(self, points, eps):
        scaler = StandardScaler()
        scaled_points = scaler.fit_transform(points)
        dbscan = DBSCAN(eps=eps)
        dbscan.fit(scaled_points)
        plane_points = []
        for i in range(len(points)):
            if dbscan.labels_[i] == 0:
                plane_points.append(points[i])
        if len(plane_points) >= 3:
            plane_points = np.array(plane_points)
            centroid = np.mean(plane_points, axis=0)
            _, _, v = np.linalg.svd(plane_points - centroid)
            normal_vector = v[2]
            return normal_vector
        else:
            return None

    def analyze_cloud(self,k, eps):
        self.load_cloud_points()
        clusters = self.separate_clusters(k=k)
        for i, cluster in enumerate(clusters):
            normal_vector = self.fit_plane(cluster, eps)
            print(f"Cloud {i + 1} is a plane")
            if normal_vector is not None:
                if np.isclose(normal_vector[2], 0.0):
                    print("The plane is vertical.")
                else:
                    print("The plane is horizontal.")
            else:
                print(f"Cloud {i + 1} is not a plane.")



