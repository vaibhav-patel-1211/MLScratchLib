import numpy as np

class SVMSGD:
    """
    Support Vector Machine (SVM) classifier implementation using Stochastic Gradient Descent.
    
    This implementation supports both linear and non-linear classification with
    different kernel functions and uses Stochastic Gradient Descent (SGD)
    for solving the optimization problem, making it suitable for large-scale datasets.
    
    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.
    
    kernel : {'linear', 'poly', 'rbf'}, default='linear'
        Kernel function to use:
        - 'linear': Linear kernel: K(x, y) = <x, y>
        - 'poly': Polynomial kernel: K(x, y) = (gamma * <x, y> + coef0)^degree
        - 'rbf': RBF kernel: K(x, y) = exp(-gamma * ||x-y||^2)
    
    learning_rate : float, default=0.01
        Initial learning rate for gradient descent.
    
    batch_size : int, default=32
        Size of minibatches for stochastic gradient descent.
        If None, uses full batch gradient descent.
    
    n_epochs : int, default=10
        Number of passes over the training data.
    
    degree : int, default=3
        Degree of polynomial kernel function ('poly').
    
    gamma : float, default='scale'
        Kernel coefficient for 'rbf' and 'poly' kernels.
        If gamma='scale' (default), then gamma = 1 / (n_features * X.var())
        If gamma='auto', then gamma = 1 / n_features
    
    coef0 : float, default=0.0
        Independent term in kernel function for 'poly' kernel.
    
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    
    Attributes
    ----------
    weights_ : ndarray of shape (n_features,)
        Model weights for linear SVM.
    
    bias_ : float
        Model bias term.
    
    support_vectors_ : ndarray of shape (n_SV, n_features)
        Support vectors for kernel SVM.
    
    dual_coef_ : ndarray of shape (n_SV,)
        Coefficients of the support vectors in the decision function.
    """
    
    def __init__(self, C=1.0, kernel='linear', learning_rate=0.01,
                 batch_size=32, n_epochs=10, degree=3, gamma='scale',
                 coef0=0.0, tol=1e-3):
        self.C = C
        self.kernel = kernel
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        
        self.weights_ = None
        self.bias_ = None
        self.support_vectors_ = None
        self.dual_coef_ = None
        self._gamma = None
    
    def _get_gamma(self, X):
        """Compute gamma based on the input data."""
        if self.gamma == 'scale':
            return 1.0 / (X.shape[1] * (X.var() + 1e-9))
        elif self.gamma == 'auto':
            return 1.0 / X.shape[1]
        return self.gamma
    
    def _kernel(self, X1, X2):
        """Compute the kernel matrix between X1 and X2."""
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'poly':
            return (self._gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            K = np.maximum(0, X1_norm + X2_norm - 2 * np.dot(X1, X2.T))
            return np.exp(-self._gamma * K)
    
    def _compute_hinge_loss_gradient(self, X, y, idx):
        """Compute the gradient of the hinge loss for the given samples."""
        if self.kernel == 'linear':
            margin = y[idx] * (np.dot(X[idx], self.weights_) + self.bias_)
            mask = margin < 1
            
            grad_w = self.weights_  # L2 regularization gradient
            grad_w -= self.C * np.sum(mask.reshape(-1, 1) * (y[idx].reshape(-1, 1) * X[idx]), axis=0)
            
            grad_b = -self.C * np.sum(mask * y[idx])
            
            return grad_w, grad_b
        else:
            # For kernel SVM, we update the dual coefficients
            K = self._kernel(X[idx], self.support_vectors_)
            margin = y[idx] * (np.dot(K, self.dual_coef_) + self.bias_)
            mask = margin < 1
            
            grad_alpha = np.zeros_like(self.dual_coef_)
            for i in range(len(idx)):
                if mask[i]:
                    grad_alpha += y[idx[i]] * K[i]
            
            grad_b = -self.C * np.sum(mask * y[idx])
            
            return grad_alpha, grad_b
    
    def _init_params(self, X):
        """Initialize model parameters."""
        if self.kernel == 'linear':
            self.weights_ = np.zeros(X.shape[1])
            self.bias_ = 0.0
        else:
            self.support_vectors_ = X.copy()
            self.dual_coef_ = np.zeros(len(X))
            self.bias_ = 0.0
    
    def fit(self, X, y):
        """Fit the SVM model using stochastic gradient descent.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.array(X)
        y = np.array(y)
        
        # Convert labels to -1, 1
        y = np.where(y <= 0, -1, 1)
        
        self._gamma = self._get_gamma(X)
        self._init_params(X)
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Training loop
        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            
            if self.batch_size is None:
                batch_indices = [indices]
            else:
                batch_indices = [indices[i:i + self.batch_size]
                                for i in range(0, n_samples, self.batch_size)]
            
            # Adaptive learning rate
            current_lr = self.learning_rate / (1 + epoch * 0.1)
            
            for batch_idx in batch_indices:
                if self.kernel == 'linear':
                    grad_w, grad_b = self._compute_hinge_loss_gradient(X, y, batch_idx)
                    self.weights_ -= current_lr * grad_w
                    self.bias_ -= current_lr * grad_b
                else:
                    grad_alpha, grad_b = self._compute_hinge_loss_gradient(X, y, batch_idx)
                    self.dual_coef_ += current_lr * grad_alpha
                    self.bias_ -= current_lr * grad_b
                    
                    # Project dual coefficients to satisfy constraints
                    self.dual_coef_ = np.clip(self.dual_coef_, 0, self.C)
        
        if self.kernel != 'linear':
            # Keep only support vectors
            sv = np.abs(self.dual_coef_) > 1e-5
            self.support_vectors_ = self.support_vectors_[sv]
            self.dual_coef_ = self.dual_coef_[sv]
        
        return self
    
    def predict(self, X):
        """Perform classification on samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class labels for samples in X.
        """
        X = np.array(X)
        decision = self._decision_function(X)
        return np.sign(decision)
    
    def _decision_function(self, X):
        """Compute the decision function of X."""
        if self.kernel == 'linear':
            return np.dot(X, self.weights_) + self.bias_
        else:
            kernel = self._kernel(X, self.support_vectors_)
            return np.dot(kernel, self.dual_coef_) + self.bias_
    
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
        y = np.where(y <= 0, -1, 1)
        return np.mean(y_pred == y)