import numpy as np

class KMeans:
    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.max_iter = max_iter

    def fit(self, X):
        X = np.array(X)
        n_samples, n_features = X.shape

        # 1. initialize centroids randomly
        random_idx = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_idx]

        for _ in range(self.max_iter):
            # 2. assign clusters
            labels = []
            for x in X:
                distances = [np.linalg.norm(x - c) for c in self.centroids]
                labels.append(np.argmin(distances))
            labels = np.array(labels)

            # 3. update centroids
            new_centroids = []
            for i in range(self.k):
                cluster_points = X[labels == i]
                new_centroids.append(cluster_points.mean(axis=0))
            new_centroids = np.array(new_centroids)

            # 4. check convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        self.labels = labels
        return self

    def predict(self, X):
        X = np.array(X)
        labels = []

        for x in X:
            distances = [np.linalg.norm(x - c) for c in self.centroids]
            labels.append(np.argmin(distances))

        return np.array(labels)
