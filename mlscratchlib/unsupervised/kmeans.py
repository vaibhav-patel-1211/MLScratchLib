import numpy as np

class KMeans:
    """
    K-Means Clustering implementation from scratch using NumPy.
    Includes K-Means++ initialization and WCSS calculation.
    """

    def __init__(self, n_clusters=3, max_iter=300, random_state=None, init='kmeans++'):
        """
        Initialize the K-Means object.

        Args:
            n_clusters (int): The number of clusters (K).
            max_iter (int): Maximum number of iterations for the K-Means algorithm.
            random_state (int or None): Seed for random number generation.
            init (str): Centroid initialization method ('random' or 'kmeans++').
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.init = init
        self.centroids = None
        self.labels = None

    def _initialize_centroids(self, X):
        """
        Initializes centroids using the specified method ('random' or 'kmeans++').
        """
        n_samples, n_features = X.shape
        np.random.seed(self.random_state)

        if self.init == 'random':
            # Select K random indices and use those points as initial centroids
            random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            self.centroids = X[random_indices]

        elif self.init == 'kmeans++':
            # 1. Select the first centroid uniformly at random
            self.centroids = np.empty((self.n_clusters, n_features))
            first_index = np.random.choice(n_samples)
            self.centroids[0] = X[first_index]

            for i in range(1, self.n_clusters):
                # Calculate the distance of each point to the nearest existing centroid
                distances_sq = self._closest_distance_sq(X)

                # 2. Select a new centroid with probability proportional to D(x)^2
                # The total sum of distances_sq is used for normalization
                probabilities = distances_sq / np.sum(distances_sq)

                # 3. Choose the next centroid based on the calculated probabilities
                next_centroid_index = np.random.choice(n_samples, p=probabilities)
                self.centroids[i] = X[next_centroid_index]
        else:
            raise ValueError("Initialization method must be 'random' or 'kmeans++'")

    def _closest_distance_sq(self, X):
        """
        Calculates the squared distance of each point in X to its nearest existing centroid.
        Used for K-Means++ initialization.
        """
        # Calculate squared Euclidean distances from all points to all current centroids
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

        # Find the minimum distance (to the closest centroid) for each data point
        min_distances = np.min(distances, axis=1)
        return min_distances**2

    def _assign_clusters(self, X):
        """
        Assignment Step (E-Step): Assigns each data point to the closest centroid.

        Returns:
            np.ndarray: An array of cluster indices (labels) for each sample.
        """
        # Calculate squared Euclidean distance (L2 norm) between all points and all centroids
        # X[:, np.newaxis] -> (n_samples, 1, n_features)
        # self.centroids -> (n_clusters, n_features)
        # Result of difference -> (n_samples, n_clusters, n_features)
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

        # Find the index of the minimum distance for each sample
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        """
        Update Step (M-Step): Recalculates the centroids as the mean of the assigned points.

        Returns:
            np.ndarray: The new centroid positions.
        """
        new_centroids = np.zeros_like(self.centroids)
        for i in range(self.n_clusters):
            # Select all points belonging to cluster i
            cluster_points = X[labels == i]

            # If the cluster is not empty, calculate the mean
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # Handle empty cluster: keep the old centroid position
                # or re-initialize it (keeping old is simpler for interview)
                new_centroids[i] = self.centroids[i]

        return new_centroids

    def fit(self, X):
        """
        Executes the K-Means algorithm until convergence or max_iter is reached.

        Args:
            X (np.ndarray): The dataset (n_samples, n_features).
        """
        # Step 0: Initialization
        self._initialize_centroids(X)

        for i in range(self.max_iter):
            # E-Step: Assign clusters
            self.labels = self._assign_clusters(X)

            # M-Step: Update centroids
            new_centroids = self._update_centroids(X, self.labels)

            # Check for convergence: stop if centroids haven't moved
            if np.allclose(self.centroids, new_centroids):
                print(f"Converged after {i+1} iterations.")
                self.centroids = new_centroids
                break

            self.centroids = new_centroids

        # Final labels after max_iter (if not converged)
        self.labels = self._assign_clusters(X)
        return self

    def predict(self, X):
        """
        Predicts the cluster labels for new data points.
        """
        return self._assign_clusters(X)

    def wcss(self, X):
        """
        Calculates the Within-Cluster Sum of Squares (WCSS).
        WCSS is the sum of squared distances of samples to their closest centroid.
        """
        if self.labels is None:
             raise RuntimeError("The model must be fitted before calculating WCSS.")

        total_wcss = 0
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]

            # Distance of points in cluster i to its centroid i
            squared_distances = np.sum(np.square(cluster_points - self.centroids[i]))
            total_wcss += squared_distances

        return total_wcss
