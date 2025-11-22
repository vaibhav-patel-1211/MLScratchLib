import numpy as np

class SVM:
    """
    Support Vector Machine (SVM) classifier implementation.
    
    This implementation supports both linear and non-linear classification with
    different kernel functions and uses Sequential Minimal Optimization (SMO)
    for solving the quadratic optimization problem.
    
    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.
    
    kernel : {'linear', 'poly', 'rbf'}, default='rbf'
        Kernel function to use:
        - 'linear': Linear kernel: K(x, y) = <x, y>
        - 'poly': Polynomial kernel: K(x, y) = (gamma * <x, y> + coef0)^degree
        - 'rbf': RBF kernel: K(x, y) = exp(-gamma * ||x-y||^2)
    
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
    
    max_iter : int, default=1000
        Maximum number of iterations for optimization.
    
    Attributes
    ----------
    support_vectors_ : ndarray of shape (n_SV, n_features)
        Support vectors.
    
    dual_coef_ : ndarray of shape (n_SV,)
        Coefficients of the support vectors in the decision function.
    
    intercept_ : float
        Constants in decision function.
    """
    
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, tol=1e-3, max_iter=1000):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        
        self.support_vectors_ = None
        self.dual_coef_ = None
        self.intercept_ = None
        self._gamma = None
        self.X_train_ = None
        self.y_train_ = None
    
    def _get_gamma(self, X):
        """Compute gamma based on the input data.
        - small gamma : smoother decision boundary
        - large gamma : more complex and tighter decision boundary

        gamma = 1/(no. of features * variance of X)
            - tolerance 1e-9 is added to avoid division by 0.
        """
        if self.gamma == 'scale':
            return 1.0 / (X.shape[1] * (X.var() + 1e-9))
        elif self.gamma == 'auto':
            return 1.0 / X.shape[1]
        return self.gamma
    
    def _kernel(self, X1, X2):
        """Compute the kernel matrix between X1 and X2."""
        if self.kernel == 'linear':
            """
            K(x,x)=x⊤ . x′
            """
            return np.dot(X1, X2.T)
        elif self.kernel == 'poly':
            """
            K(x,x′)=(γ⋅x⊤x′ +coef0)**d
            """
            return (self._gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            # Using broadcasting to compute pairwise squared Euclidean distances
            X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
            K = np.maximum(0, X1_norm + X2_norm - 2 * np.dot(X1, X2.T))
            return np.exp(-self._gamma * K)
    
    def _smo(self, X, y):
        """Sequential Minimal Optimization algorithm for training SVM."""
        n_samples = X.shape[0]
        alpha = np.zeros(n_samples)
        b = 0.0
        unchanged_count = 0
        
        # Compute kernel matrix
        K = self._kernel(X, X)
        
        for _ in range(self.max_iter):
            alpha_prev = alpha.copy()
            num_changed_alphas = 0
            
            for i in range(n_samples):
                # Compute model output

                # prediction
                f_i = np.sum(alpha * y * K[i]) + b
    
                # error
                E_i = f_i - y[i]

                # Karush-Kuhn-Tucker (KKT) condition.
                kkt_i = y[i] * E_i
                
                if (kkt_i < -self.tol and alpha[i] < self.C) or \
                   (kkt_i > self.tol and alpha[i] > 0):
                    
                    # Select second alpha randomly
                    j = i
                    while j == i:
                        j = np.random.randint(n_samples)
                    
                    # Compute error
                    f_j = np.sum(alpha * y * K[j]) + b
                    E_j = f_j - y[j]
                    
                    # Save old alphas
                    alpha_i_old = alpha[i]
                    alpha_j_old = alpha[j]
                    
                    # Compute L and H
                    if y[i] != y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta
                    eta = 2 * K[i,j] - K[i,i] - K[j,j]
                    if eta >= 0:
                        continue
                    
                    # Update alpha_j
                    alpha[j] = alpha_j_old - (y[j] * (E_i - E_j)) / eta
                    alpha[j] = min(H, max(L, alpha[j]))
                    
                    if abs(alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update alpha_i
                    alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha[j])
                    
                    # Update threshold
                    b1 = b - E_i - y[i] * (alpha[i] - alpha_i_old) * K[i,i] - \
                         y[j] * (alpha[j] - alpha_j_old) * K[i,j]
                    b2 = b - E_j - y[i] * (alpha[i] - alpha_i_old) * K[i,j] - \
                         y[j] * (alpha[j] - alpha_j_old) * K[j,j]
                    
                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
            
            # Check convergence
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.tol:
                unchanged_count += 1
            else:
                unchanged_count = 0
            
            # Exit if solution is stable for several iterations
            if unchanged_count >= 3:
                break
        
        # Get support vectors
        sv = alpha > 1e-5
        self.support_vectors_ = X[sv]
        self.dual_coef_ = alpha[sv] * y[sv]
        self.intercept_ = b
        
        return self
    
    def fit(self, X, y):
        """Fit the SVM model according to the given training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values (class labels in classification).
        
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.array(X)
        y = np.array(y)
        
        # Convert labels to -1, 1
        y = np.where(y <= 0, -1, 1)
        
        self.X_train_ = X
        self.y_train_ = y
        self._gamma = self._get_gamma(X)
        
        return self._smo(X, y)
    
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
        kernel = self._kernel(X, self.support_vectors_)
        return np.dot(kernel, self.dual_coef_) + self.intercept_
    
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


class SVR(SVM):
    """
    Support Vector Regression (SVR) implementation.
    
    This implementation inherits from SVM and modifies the optimization
    problem to perform regression instead of classification.
    
    Additional Parameters
    ----------
    epsilon : float, default=0.1
        Epsilon in the epsilon-SVR model. It specifies the epsilon-tube
        within which no penalty is associated in the training loss function
        with points predicted within a distance epsilon from the actual value.
    
    Other parameters and attributes are the same as SVM.
    """
    
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, epsilon=0.1, tol=1e-3, max_iter=1000):
        super().__init__(C=C, kernel=kernel, degree=degree, gamma=gamma,
                        coef0=coef0, tol=tol, max_iter=max_iter)
        self.epsilon = epsilon
    
    def _smo(self, X, y):
        """Sequential Minimal Optimization algorithm for training SVR."""
        n_samples = X.shape[0]
        # Initialize dual variables (alpha - alpha*)
        alpha = np.zeros(2 * n_samples)  # First n: alpha, Last n: alpha*
        b = 0.0
        
        # Compute kernel matrix
        K = self._kernel(X, X)
        
        for _ in range(self.max_iter):
            alpha_prev = alpha.copy()
            
            for i in range(2 * n_samples):
                idx = i % n_samples
                is_alpha = i < n_samples
                
                # Compute output
                f_i = np.sum((alpha[:n_samples] - alpha[n_samples:]) * K[idx]) + b
                
                # Compute error
                if is_alpha:
                    E_i = f_i - (y[idx] + self.epsilon)
                else:
                    E_i = f_i - (y[idx] - self.epsilon)
                
                # Check KKT conditions
                if (is_alpha and ((E_i < -self.tol and alpha[i] < self.C) or 
                    (E_i > self.tol and alpha[i] > 0))) or \
                   (not is_alpha and ((E_i > self.tol and alpha[i] < self.C) or 
                    (E_i < -self.tol and alpha[i] > 0))):
                    
                    # Select second alpha randomly
                    j = i
                    while j == i:
                        j = np.random.randint(2 * n_samples)
                    
                    # Compute errors
                    idx_j = j % n_samples
                    is_alpha_j = j < n_samples
                    
                    f_j = np.sum((alpha[:n_samples] - alpha[n_samples:]) * K[idx_j]) + b
                    
                    if is_alpha_j:
                        E_j = f_j - (y[idx_j] + self.epsilon)
                    else:
                        E_j = f_j - (y[idx_j] - self.epsilon)
                    
                    # Save old alphas
                    alpha_i_old = alpha[i]
                    alpha_j_old = alpha[j]
                    
                    # Compute L and H
                    if (is_alpha and is_alpha_j) or (not is_alpha and not is_alpha_j):
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])
                    else:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    
                    if L == H:
                        continue
                    
                    # Compute eta
                    eta = 2 * K[idx,idx_j] - K[idx,idx] - K[idx_j,idx_j]
                    if eta >= 0:
                        continue
                    
                    # Update alpha_j
                    alpha[j] = alpha_j_old - (E_i - E_j) / eta
                    alpha[j] = min(H, max(L, alpha[j]))
                    
                    if abs(alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update alpha_i
                    alpha[i] = alpha_i_old + (alpha_j_old - alpha[j])
                    
                    # Update threshold
                    if 0 < alpha[i] < self.C:
                        b_i = E_i + (alpha[i] - alpha_i_old) * K[idx,idx] + \
                             (alpha[j] - alpha_j_old) * K[idx,idx_j]
                        b = b_i
                    elif 0 < alpha[j] < self.C:
                        b_j = E_j + (alpha[i] - alpha_i_old) * K[idx,idx_j] + \
                             (alpha[j] - alpha_j_old) * K[idx_j,idx_j]
                        b = b_j
                    else:
                        b_i = E_i + (alpha[i] - alpha_i_old) * K[idx,idx] + \
                             (alpha[j] - alpha_j_old) * K[idx,idx_j]
                        b_j = E_j + (alpha[i] - alpha_i_old) * K[idx,idx_j] + \
                             (alpha[j] - alpha_j_old) * K[idx_j,idx_j]
                        b = (b_i + b_j) / 2
            
            # Check convergence
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self.tol:
                break
        
        # Get support vectors
        alpha_diff = alpha[:n_samples] - alpha[n_samples:]
        sv = np.abs(alpha_diff) > 1e-5
        self.support_vectors_ = X[sv]
        self.dual_coef_ = alpha_diff[sv]
        self.intercept_ = b
        
        return self
    
    def fit(self, X, y):
        """Fit the SVR model according to the given training data.
        
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
        
        self.X_train_ = X
        self.y_train_ = y
        self._gamma = self._get_gamma(X)
        
        return self._smo(X, y)
    
    def predict(self, X):
        """Predict using the SVR model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.
        """
        return self._decision_function(np.array(X))
    
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