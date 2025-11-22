import numpy as np
from collections import Counter

class KNNClassifier:
    """
    K-Nearest Neighbors (KNN) classifier implementation.

    This implementation supports both classification and regression tasks with
    different distance metrics (Euclidean, Manhattan) and voting schemes.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use for prediction.

    metric : {'euclidean', 'manhattan'}, default='euclidean'
        Distance metric to use for neighbor computation:
        - 'euclidean': Euclidean distance (L2 norm)
        - 'manhattan': Manhattan distance (L1 norm)

    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction:
        - 'uniform': All points in each neighborhood are weighted equally
        - 'distance': Points are weighted by the inverse of their distance

    Attributes
    ----------
    X_train_ : ndarray of shape (n_samples, n_features)
        Training data.

    y_train_ : ndarray of shape (n_samples,)
        Target values for training data.
    """

    def __init__(self, n_neighbors=5, metric='euclidean', weights='uniform'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.X_train_ = None
        self.y_train_ = None

    def _euclidean_distance(self, x1, x2):
        """Compute Euclidean distance between two points."""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _manhattan_distance(self, x1, x2):
        """Compute Manhattan distance between two points."""
        return np.sum(np.abs(x1 - x2))

    def _get_distances(self, X):
        """Compute distances between test samples and training samples."""
        distances = []
        for x in X:
            if self.metric == 'euclidean':
                dist = [self._euclidean_distance(x, x_train) for x_train in self.X_train_]
            else:  # manhattan
                dist = [self._manhattan_distance(x, x_train) for x_train in self.X_train_]
            distances.append(dist)
        return np.array(distances)
    def _get_weights(self, distances):
        """Compute weights for the neighbors based on their distances."""
        if self.weights == 'uniform':
            return np.ones(distances.shape)
        else:  # distance weighting
            # Add small constant to avoid division by zero
            return 1 / (distances + 1e-10)

    def fit(self, X, y):
        """Fit the KNN model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        self.X_train_ = np.array(X)
        self.y_train_ = np.array(y)
        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        X = np.array(X)
        distances = self._get_distances(X)
        weights = self._get_weights(distances)
        y_pred = []

        for i, dist in enumerate(distances):
            # Get indices of k nearest neighbors
            k_indices = np.argsort(dist)[:self.n_neighbors]
            k_nearest_labels = self.y_train_[k_indices]
            k_weights = weights[i, k_indices]

            # Weighted voting
            if len(np.unique(k_nearest_labels)) == 1:
                y_pred.append(k_nearest_labels[0])
            else:
                # Weight the votes by distance
                weighted_votes = {}
                for label, weight in zip(k_nearest_labels, k_weights):
                    weighted_votes[label] = weighted_votes.get(label, 0) + weight

                # Select the class with highest weighted votes
                y_pred.append(max(weighted_votes.items(), key=lambda x: x[1])[0])

        return np.array(y_pred)

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) with respect to y.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class KNNRegressor(KNNClassifier):
    """
    K-Nearest Neighbors regressor implementation.

    This implementation inherits from KNNClassifier and overrides the predict
    method to perform regression instead of classification.

    Parameters and attributes are the same as KNNClassifier.
    """

    def predict(self, X):
        """Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values.
        """
        X = np.array(X)
        distances = self._get_distances(X)
        weights = self._get_weights(distances)
        y_pred = []

        for i, dist in enumerate(distances):
            # Get indices of k nearest neighbors
            k_indices = np.argsort(dist)[:self.n_neighbors]
            k_nearest_targets = self.y_train_[k_indices]
            k_weights = weights[i, k_indices]

            # Weighted average for regression
            y_pred.append(np.average(k_nearest_targets, weights=k_weights))

        return np.array(y_pred)

    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True values for X.

        Returns
        -------
        score : float
            R^2 score.
        """
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - (u/v)
