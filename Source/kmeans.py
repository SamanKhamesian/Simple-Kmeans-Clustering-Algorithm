import random
import sys

import numpy as np


class KMeans:
    def __init__(self, k, max_iter=1000, epsilon=1e-5):
        self.k = k
        self.__max_iter = max_iter
        self.__epsilon = epsilon
        self.__centers = None
        self.__cluster_id = None

    # Return a list of cluster id for all samples
    def get_labels(self):
        return self.__cluster_id

    # Return a list of the centers of each cluster
    def get_centers(self):
        return self.__centers

    # Calculate the Euclidean distance for the two given points
    @staticmethod
    def __get_distance(x_1, x_2):
        return np.linalg.norm(x_1 - x_2)

    # Create k clusters with their centers
    def __init_clusters(self):
        clusters = list()
        for i in range(self.k):
            new_cluster = list()
            new_cluster.append(self.__centers[i])
            clusters.append(new_cluster)
        return clusters

    # Find the closest cluster for a single point
    def __find_closest_cluster(self, sample):
        dists = [self.__get_distance(c, sample) for c in self.__centers]
        min_dist = min(dists)
        return dists.index(min_dist)

    # Assign every sample to its proper cluster
    def __group_samples(self, clusters, x):
        for i, sample in enumerate(x):
            c_index = self.__find_closest_cluster(sample)
            self.__cluster_id[i] = c_index
            clusters[c_index].append(sample)
        return clusters

    # Recompute the center of each cluster
    def __update_centers(self, error, clusters):
        for i, c in enumerate(clusters):
            c_mean = np.mean(c, axis=0)
            error = min(error, max(abs(c_mean - self.__centers[i])))
            self.__centers[i] = c_mean
        return error

    def fit(self, x):
        error = sys.maxsize
        counter = 0
        n_sample = x.shape[0]
        self.__centers = random.sample(list(x), k=self.k)
        self.__cluster_id = np.zeros(n_sample, dtype=int)

        while error > self.__epsilon and counter < self.__max_iter:
            clusters = self.__init_clusters()
            clusters = self.__group_samples(clusters, x)
            error = self.__update_centers(error, clusters)
