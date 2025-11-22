import numpy as np
from .linear_regression import LinearRegression

class LassoRegression(LinearRegression):
    """
    Lasso Regression implementation using gradient descent optimization.
    
    Lasso regression adds L1 regularization to linear regression, which can
    lead to sparse models where some coefficients become exactly zero. This
    makes it useful for feature selection.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the L1 term, controlling regularization strength.
        alpha = 0 is equivalent to unregularized linear regression.
        Higher values specify stronger regularization.
    
    learning_rate : float, default=0.01
        The step size for gradient descent optimization.
    
    n_iterations : int, default=1000
        Maximum number of iterations for gradient descent.
    
    batch_size : int, default=None
        Size of minibatches for stochastic gradient descent.
        If None, uses batch gradient descent.
    
    Attributes
    ----------
    weights_ : ndarray of shape (n_features,)
        Coefficients for the linear regression model.
    
    bias_ : float
        Intercept (bias) term for the linear regression model.
    
    cost_history_ : list
        History of cost function values during training.
    """
    
    def __init__(self, alpha=1.0, learning_rate=0.01, n_iterations=1000, batch_size=None):
        super().__init__(
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            batch_size=batch_size,
            regularization='l1',
            alpha=alpha
        )