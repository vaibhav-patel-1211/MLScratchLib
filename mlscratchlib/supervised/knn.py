import numpy as np

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        X = np.array(X)
        predictions = []

        for x in X:
            # 1. compute distances
            distances = []
            for x_train in self.X_train:
                distances.append(self._euclidean_distance(x, x_train))

            # 2. k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]

            # 3. majority voting
            values, counts = np.unique(k_labels, return_counts=True)
            predicted_label = values[np.argmax(counts)]

            predictions.append(predicted_label)

        return np.array(predictions)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

class KNNRegressor:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        X = np.array(X)
        predictions = []

        for x in X:
            # 1. compute distances
            distances = []
            for x_train in self.X_train:
                distances.append(self._euclidean_distance(x, x_train))

            # 2. k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_values = self.y_train[k_indices]

            # 3. mean for regression
            predicted_value = np.mean(k_values)
            predictions.append(predicted_value)

        return np.array(predictions)

    def score(self, X, y):
        y_pred = self.predict(X)
        num = np.sum((y - y_pred) ** 2)
        den = np.sum((y - np.mean(y)) ** 2)
        return 1 - (num / den)
