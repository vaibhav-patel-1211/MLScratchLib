import numpy as np
from collections import Counter

class Node:
    """A decision tree node.

    Parameters
    ----------
    feature_idx : int or None
        The index of the feature used for splitting at this node.
        None for leaf nodes.
    threshold : float or None
        The threshold value for the split.
        None for leaf nodes.
    value : float or dict
        The predicted value (for regression) or class probabilities (for classification)
        at this node.
    left : Node or None
        The left child node.
    right : Node or None
        The right child node.
    """
    def __init__(self, feature_idx=None, threshold=None, value=None, left=None, right=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right

class BaseDecisionTree:
    """Base class for decision trees.

    Parameters
    ----------
    max_depth : int or None, default=None
        The maximum depth of the tree. If None, nodes are expanded until all
        leaves are pure or contain less than min_samples_split samples.

    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
    """
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None

    def _is_leaf(self, depth, n_samples, impurity):
        """Check if a node should be a leaf node."""
        return (
            (self.max_depth is not None and depth >= self.max_depth) or
            n_samples < self.min_samples_split or
            n_samples < 2 * self.min_samples_leaf or
            impurity <= self.min_impurity_decrease
        )

    def _split(self, X, y, feature_idx, threshold):
        """Split dataset based on feature and threshold."""
        mask = X[:, feature_idx] <= threshold
        return (
            X[mask], X[~mask],
            y[mask], y[~mask]
        )

    def predict(self, X):
        """Predict target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted values.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """Traverse tree for prediction."""
        if node.feature_idx is None:  # leaf node
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class DecisionTreeClassifier(BaseDecisionTree):
    """A decision tree classifier.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split.

    max_depth : int or None, default=None
        The maximum depth of the tree.

    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
    """
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0.0):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease
        )
        self.criterion = criterion

    def _calc_impurity(self, y):
        """Calculate impurity (gini or entropy) of a node."""
        if len(y) == 0:
            return 0

        probs = np.array([count/len(y) for count in Counter(y).values()])

        if self.criterion == "gini":
            return 1 - np.sum(probs ** 2)
        else:  # entropy
            return -np.sum(probs * np.log2(probs + 1e-10))

    def _calc_leaf_value(self, y):
        """Calculate the most common class for a leaf node."""
        return max(Counter(y).items(), key=lambda x: x[1])[0]

    def _find_best_split(self, X, y):
        """Find the best split for a node."""
        n_samples, n_features = X.shape
        best_gain = -1
        best_feature = None
        best_threshold = None

        current_impurity = self._calc_impurity(y)

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self._split(X, y, feature_idx, threshold)

                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                gain = current_impurity - (
                    len(y_left) * self._calc_impurity(y_left) +
                    len(y_right) * self._calc_impurity(y_right)
                ) / n_samples

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples = len(y)
        impurity = self._calc_impurity(y)

        if self._is_leaf(depth, n_samples, impurity):
            return Node(value=self._calc_leaf_value(y))

        feature_idx, threshold, gain = self._find_best_split(X, y)

        if feature_idx is None or gain <= self.min_impurity_decrease:
            return Node(value=self._calc_leaf_value(y))

        X_left, X_right, y_left, y_right = self._split(X, y, feature_idx, threshold)

        left = self._build_tree(X_left, y_left, depth + 1)
        right = self._build_tree(X_right, y_right, depth + 1)

        return Node(feature_idx=feature_idx, threshold=threshold,
                   value=self._calc_leaf_value(y), left=left, right=right)

    def fit(self, X, y):
        """Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        self.root = self._build_tree(np.array(X), np.array(y))
        return self

class DecisionTreeRegressor(BaseDecisionTree):
    """A decision tree regressor.

    Parameters
    ----------
    max_depth : int or None, default=None
        The maximum depth of the tree.

    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
    """
    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0.0):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease
        )

    def _calc_impurity(self, y):
        """Calculate impurity (MSE) of a node."""
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def _calc_leaf_value(self, y):
        """Calculate the mean value for a leaf node."""
        return np.mean(y)

    def _find_best_split(self, X, y):
        """Find the best split for a node."""
        n_samples, n_features = X.shape
        best_gain = -1
        best_feature = None
        best_threshold = None

        current_impurity = self._calc_impurity(y)

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self._split(X, y, feature_idx, threshold)

                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                gain = current_impurity - (
                    len(y_left) * self._calc_impurity(y_left) +
                    len(y_right) * self._calc_impurity(y_right)
                ) / n_samples

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples = len(y)
        impurity = self._calc_impurity(y)

        if self._is_leaf(depth, n_samples, impurity):
            return Node(value=self._calc_leaf_value(y))

        feature_idx, threshold, gain = self._find_best_split(X, y)

        if feature_idx is None or gain <= self.min_impurity_decrease:
            return Node(value=self._calc_leaf_value(y))

        X_left, X_right, y_left, y_right = self._split(X, y, feature_idx, threshold)

        left = self._build_tree(X_left, y_left, depth + 1)
        right = self._build_tree(X_right, y_right, depth + 1)

        return Node(feature_idx=feature_idx, threshold=threshold,
                   value=self._calc_leaf_value(y), left=left, right=right)

    def fit(self, X, y):
        """Build a decision tree regressor from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        self.root = self._build_tree(np.array(X), np.array(y))
        return self
