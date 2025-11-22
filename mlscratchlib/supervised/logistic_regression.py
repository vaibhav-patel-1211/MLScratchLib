import numpy as np

class LogisticRegression:
    """
    Logistic Regression implementation using gradient descent optimization.

    This implementation supports both binary and multiclass classification using
    one-vs-rest strategy, with options for batch and stochastic gradient descent,
    and regularization.

    Parameters
    ----------
    learning_rate : float, default=0.01
        The step size for gradient descent optimization.

    n_iterations : int, default=1000
        Maximum number of iterations for gradient descent.

    batch_size : int, default=None
        Size of minibatches for stochastic gradient descent.
        If None, uses batch gradient descent.

    regularization : {'none', 'l1', 'l2'}, default='none'
        The type of regularization to use:
        - 'none': No regularization
        - 'l1': L1 regularization (Lasso)
        - 'l2': L2 regularization (Ridge)

    alpha : float, default=0.01
        Regularization strength. Only used when regularization is 'l1' or 'l2'.

    Attributes
    ----------
    weights_ : ndarray of shape (n_classes, n_features)
        Coefficients for the logistic regression model.
        For binary classification, n_classes = 1.

    bias_ : ndarray of shape (n_classes,)
        Intercept (bias) terms for the logistic regression model.
        For binary classification, n_classes = 1.

    classes_ : ndarray of shape (n_classes,)
        Unique class labels.

    cost_history_ : list
        History of cost function values during training.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=None,
                 regularization='none', alpha=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.regularization = regularization
        self.alpha = alpha
        self.weights_ = None
        self.bias_ = None
        self.classes_ = None
        self.cost_history_ = []

    def _sigmoid(self, z):
        """Apply sigmoid activation function for binary classification."""
        # Clip z to avoid overflow in exp
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def _softmax(self, z):
        """Apply softmax activation function for multiclass classification."""
        # Clip z to avoid overflow in exp
        z = np.clip(z, -250, 250)
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _compute_cost(self, X, y, predictions):
        """Compute the cross-entropy loss with regularization if applicable."""
        m = len(y)
        epsilon = 1e-15  # Small constant to avoid log(0)

        if len(self.classes_) == 2:
            # Binary classification
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            cost = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        else:
            # Multiclass classification
            predictions = np.clip(predictions, epsilon, 1.0)
            cost = -np.mean(np.sum(y * np.log(predictions), axis=1))

        # Add regularization term if applicable
        if self.regularization == 'l1':
            # L1 regularization (Lasso)
            reg_term = (self.alpha / m) * np.sum(np.abs(self.weights_))
        elif self.regularization == 'l2':
            # L2 regularization (Ridge)
            reg_term = (self.alpha / (2 * m)) * np.sum(self.weights_ ** 2)
        else:
            reg_term = 0

        return cost + reg_term

    def _compute_gradients(self, X, y, predictions):
        """Compute gradients for weights and bias."""
        m = len(y)
        error = predictions - y

        # Gradient for weights
        if len(self.classes_) == 2:
            # Binary classification
            dw = (1/m) * np.dot(X.T, error)
            db = (1/m) * np.sum(error)
        else:
            # Multiclass classification
            dw = (1/m) * np.dot(X.T, error).T
            db = (1/m) * np.sum(error, axis=0)

        # Add regularization gradient if applicable
        if self.regularization == 'l1':
            # L1 regularization gradient
            dw += (self.alpha / m) * np.sign(self.weights_)
        elif self.regularization == 'l2':
            # L2 regularization gradient
            dw += (self.alpha / m) * self.weights_

        return dw, db

    def fit(self, X, y):
        """Fit the logistic regression model using gradient descent.

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
        # Convert inputs to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Get unique classes and convert y to one-hot encoding
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes == 2:
            # Binary classification
            y_encoded = (y == self.classes_[1]).astype(int)
            n_classes_fit = 1  # Only need one set of weights for binary
        else:
            # Multiclass classification
            y_encoded = np.zeros((len(y), n_classes))
            for i, cls in enumerate(self.classes_):
                y_encoded[:, i] = (y == cls).astype(int)
            n_classes_fit = n_classes

        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights_ = np.zeros((n_classes_fit, n_features))
        self.bias_ = np.zeros(n_classes_fit)
        self.cost_history_ = []

        # Gradient descent
        for i in range(self.n_iterations):
            if self.batch_size is None:
                # Batch gradient descent
                z = np.dot(X, self.weights_.T) + self.bias_
                if n_classes == 2:
                    predictions = self._sigmoid(z)
                else:
                    predictions = self._softmax(z)

                cost = self._compute_cost(X, y_encoded, predictions)
                dw, db = self._compute_gradients(X, y_encoded, predictions)

                # Update parameters
                self.weights_ -= self.learning_rate * dw
                self.bias_ -= self.learning_rate * db

                self.cost_history_.append(cost)
            else:
                # Mini-batch stochastic gradient descent
                for start_idx in range(0, n_samples, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, n_samples)
                    X_batch = X[start_idx:end_idx]
                    y_batch = y_encoded[start_idx:end_idx]

                    z_batch = np.dot(X_batch, self.weights_.T) + self.bias_
                    if n_classes == 2:
                        predictions = self._sigmoid(z_batch)
                    else:
                        predictions = self._softmax(z_batch)

                    dw, db = self._compute_gradients(X_batch, y_batch, predictions)

                    # Update parameters
                    self.weights_ -= self.learning_rate * dw
                    self.bias_ -= self.learning_rate * db

                # Compute cost for the entire dataset
                z = np.dot(X, self.weights_.T) + self.bias_
                if n_classes == 2:
                    predictions = self._sigmoid(z)
                else:
                    predictions = self._softmax(z)
                cost = self._compute_cost(X, y_encoded, predictions)
                self.cost_history_.append(cost)

        return self

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict probabilities for.

        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Probability estimates for each class.
            For binary classification, returns probabilities for class 1 only.
        """
        X = np.array(X)
        z = np.dot(X, self.weights_.T) + self.bias_

        if len(self.classes_) == 2:
            # Binary classification
            proba = self._sigmoid(z)
            return np.column_stack([1 - proba, proba])
        else:
            # Multiclass classification
            return self._softmax(z)

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict class labels for.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        probas = self.predict_proba(X)
        if len(self.classes_) == 2:
            return self.classes_[(probas[:, 1] >= 0.5).astype(int)]
        else:
            return self.classes_[np.argmax(probas, axis=1)]

    def score(self, X, y):
        """Return the accuracy score of the predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True class labels (0 or 1).

        Returns
        -------
        score : float
            Accuracy score.
        """
        return np.mean(self.predict(X) == y)
