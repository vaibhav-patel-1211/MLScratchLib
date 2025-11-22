import numpy as np

class LinearRegression:
    """
    Linear Regression implementation using gradient descent optimization.
    
    This implementation supports both batch and stochastic gradient descent,
    along with regularization options.
    
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
    weights_ : ndarray of shape (n_features,)
        Coefficients for the linear regression model.
    
    bias_ : float
        Intercept (bias) term for the linear regression model.
    
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
        self.cost_history_ = []
    
    def _compute_cost(self, X, y, predictions):
        """Compute the cost (Mean Squared Error + regularization if applicable)."""
        m = len(y)
        # Calculate mean squared error
        mse = np.mean((predictions - y) ** 2)
        
        # Add regularization term if applicable
        if self.regularization == 'l1':
            # L1 regularization (Lasso)
            reg_term = (self.alpha / m) * np.sum(np.abs(self.weights_))
        elif self.regularization == 'l2':
            # L2 regularization (Ridge)
            reg_term = (self.alpha / (2 * m)) * np.sum(self.weights_ ** 2)
        else:
            reg_term = 0
            
        return mse + reg_term
    
    def _compute_gradients(self, X, y, predictions):
        """Compute gradients for weights and bias."""
        m = len(y)
        # Gradient for weights
        dw = (1/m) * np.dot(X.T, (predictions - y))
        
        # Add regularization gradient if applicable
        if self.regularization == 'l1':
            # L1 regularization gradient
            dw += (self.alpha / m) * np.sign(self.weights_)
        elif self.regularization == 'l2':
            # L2 regularization gradient
            dw += (self.alpha / m) * self.weights_
        
        # Gradient for bias
        db = (1/m) * np.sum(predictions - y)
        
        return dw, db
    
    def fit(self, X, y):
        """Fit the linear regression model using gradient descent.
        
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
        """Predict using the linear regression model.
        
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
            True values.
        
        Returns
        -------
        score : float
            R^2 score.
        """
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - (u/v)