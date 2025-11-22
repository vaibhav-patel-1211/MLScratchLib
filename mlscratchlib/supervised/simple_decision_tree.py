import numpy as np
from collections import Counter

class SimpleDecisionTree:
    """A simplified decision tree that can handle both classification and regression.
    
    Parameters
    ----------
    task : str, default='classification'
        The type of task. Either 'classification' or 'regression'.
    max_depth : int or None, default=None
        The maximum depth of the tree.
    min_samples : int, default=2
        The minimum number of samples required to split a node.
    """
    def __init__(self, task='classification', max_depth=None, min_samples=2):
        self.task = task
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None
    
    class Node:
        def __init__(self, value=None):
            self.feature = None  # which feature to split on
            self.threshold = None  # threshold value for the split
            self.value = value  # prediction value for leaf nodes
            self.left = None  # left child
            self.right = None  # right child
    
    def _calculate_leaf_value(self, y):
        """Calculate the prediction value for a leaf node."""
        if self.task == 'classification':
            # Return most common class
            return max(Counter(y).items(), key=lambda x: x[1])[0]
        else:
            # Return mean value for regression
            return np.mean(y)
    
    def _calculate_impurity(self, y):
        """Calculate node impurity (gini for classification, mse for regression)."""
        if len(y) == 0:
            return 0
            
        if self.task == 'classification':
            # Calculate Gini impurity
            proportions = np.array([count/len(y) for count in Counter(y).values()])
            return 1 - np.sum(proportions ** 2)
        else:
            # Calculate MSE
            return np.mean((y - np.mean(y)) ** 2)
    
    def _find_split(self, X, y):
        """Find the best split for a node."""
        best_gain = -float('inf')
        best_split = None
        
        n_features = X.shape[1]
        current_impurity = self._calculate_impurity(y)
        
        # Try each feature
        for feature in range(n_features):
            # Try each unique value as threshold
            for threshold in np.unique(X[:, feature]):
                # Split data
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                # Skip if split is too small
                if sum(left_mask) < self.min_samples or sum(right_mask) < self.min_samples:
                    continue
                
                # Calculate impurity decrease
                left_impurity = self._calculate_impurity(y[left_mask])
                right_impurity = self._calculate_impurity(y[right_mask])
                
                # Weighted average of child impurities
                n_left, n_right = sum(left_mask), sum(right_mask)
                gain = current_impurity - (n_left * left_impurity + n_right * right_impurity) / len(y)
                
                # Update best split if this is better
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, threshold)
        
        return best_split, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        node = self.Node()
        
        # Stop if max depth reached or node is pure
        if (self.max_depth is not None and depth >= self.max_depth) or \
           len(y) < self.min_samples or len(np.unique(y)) == 1:
            node.value = self._calculate_leaf_value(y)
            return node
        
        # Find best split
        best_split, best_gain = self._find_split(X, y)
        
        # If no good split found, make leaf node
        if best_split is None or best_gain <= 0:
            node.value = self._calculate_leaf_value(y)
            return node
        
        # Split data
        feature, threshold = best_split
        left_mask = X[:, feature] <= threshold
        
        # Create decision node
        node.feature = feature
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        
        return node
    
    def fit(self, X, y):
        """Build the decision tree.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        """
        X, y = np.array(X), np.array(y)
        self.root = self._build_tree(X, y)
        return self
    
    def predict(self, X):
        """Make predictions for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
        
        Returns
        -------
        y : array-like of shape (n_samples,)
            Predicted values
        """
        X = np.array(X)
        return np.array([self._predict_one(x) for x in X])
    
    def _predict_one(self, x):
        """Make prediction for a single sample."""
        node = self.root
        
        while node.value is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        
        return node.value