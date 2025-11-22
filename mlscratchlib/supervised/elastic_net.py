import numpy as np

class ElasticNetRegression:
    """
    Elastic Net Regression implementation.
    
    This implementation combines L1 (Lasso) and L2 (Ridge) regularization
    using gradient descent optimization.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the regularization terms.
    
    l1_ratio : float, default=0.5
        The mixing parameter, with 0 <= l1_ratio <= 1.
        - l1_ratio = 1: Same as Lasso
        - l1_ratio = 0: Same as Ridge
        - 0 < l1_ratio < 1: Combination of Lasso and Ridge
    
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    
    n_iterations : int, default=1000
        Maximum number of iterations for gradient descent.
    
    batch_size : int, default=None
        Size of minibatches for stochastic gradient descent.
        If None, use batch gradient descent.
    
    Attributes
    ----------
    weights_ : ndarray of shape (n_features,)
        Coefficients for the features.
    
    bias_ : float
        The bias term.
    
    cost_history_ : list
        History of cost function values during training.
    """
    
    def __init__(self, alpha=1.0, l1_ratio=0.5, learning_rate=0.01,
                 n_iterations=1000, batch_size=None):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.weights_ = None
        self.bias_ = None
        self.cost_history_ = None
    
    def _compute_cost(self, X, y, predictions):
        """Compute the cost function with L1 and L2 regularization."""
        n_samples = len(y)
        mse = np.mean((predictions - y) ** 2)
        
        # L1 regularization term
        l1_term = self.alpha * self.l1_ratio * np.sum(np.abs(self.weights_))
        
        # L2 regularization term
        l2_term = 0.5 * self.alpha * (1 - self.l1_ratio) * np.sum(self.weights_ ** 2)
        
        return mse + (l1_term + l2_term) / n_samples
    
    def _compute_gradients(self, X, y, predictions):
        """Compute gradients for both weights and bias."""
        n_samples = len(y)
        error = predictions - y
        
        # Gradient of MSE
        dw = (2/n_samples) * np.dot(X.T, error)
        
        # Add L1 regularization gradient
        l1_grad = self.alpha * self.l1_ratio * np.sign(self.weights_)
        
        # Add L2 regularization gradient
        l2_grad = self.alpha * (1 - self.l1_ratio) * self.weights_
        
        # Combine gradients
        dw += (l1_grad + l2_grad) / n_samples
        
        # Bias gradient (no regularization for bias)
        db = (2/n_samples) * np.sum(error)
        
        return dw, db
    
    def fit(self, X, y):
        """Fit the Elastic Net regression model using gradient descent.
        
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
        
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights_ = np.zeros(n_features)
        self.bias_ = 0
        self.cost_history_ = []
        
        # Gradient descent
        for i in range(self.n_iterations):
            if self.batch_size is None:
                # Batch gradient descent
                predictions = np.dot(X, self.weights_) + self.bias_
                cost = self._compute_cost(X, y, predictions)
                dw, db = self._compute_gradients(X, y, predictions)
                
                # Update parameters
                self.weights_ -= self.learning_rate * dw
                self.bias_ -= self.learning_rate * db
                
                self.cost_history_.append(cost)
            else:
                # Mini-batch stochastic gradient descent
                for start_idx in range(0, n_samples, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, n_samples)
                    X_batch = X[start_idx:end_idx]
                    y_batch = y[start_idx:end_idx]
                    
                    predictions = np.dot(X_batch, self.weights_) + self.bias_
                    dw, db = self._compute_gradients(X_batch, y_batch, predictions)
                    
                    # Update parameters
                    self.weights_ -= self.learning_rate * dw
                    self.bias_ -= self.learning_rate * db
                
                # Compute cost for the entire dataset
                predictions = np.dot(X, self.weights_) + self.bias_
                cost = self._compute_cost(X, y, predictions)
                self.cost_history_.append(cost)
        
        return self
    
    def predict(self, X):
        """Predict using the Elastic Net regression model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values.
        """
        X = np.array(X)
        return np.dot(X, self.weights_) + self.bias_
    
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