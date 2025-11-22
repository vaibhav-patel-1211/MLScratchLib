import numpy as np

class SVD:
    """
    Singular Value Decomposition (SVD) implementation.
    
    SVD decomposes a matrix X into three matrices U, Σ, and V such that X = UΣV^T,
    where U and V are orthogonal matrices and Σ is a diagonal matrix of singular values.
    This implementation supports truncated SVD for dimensionality reduction.
    
    Parameters
    ----------
    n_components : int or float, default=None
        Number of components to keep:
        - If n_components is None, keep all components
        - If int, n_components must be <= min(n_samples, n_features)
        - If float between 0 and 1, select the number of components such that
          the amount of variance that needs to be explained is greater than
          the percentage specified
    
    Attributes
    ----------
    singular_values_ : ndarray of shape (n_components,)
        The singular values, sorted in descending order.
    
    explained_variance_ : ndarray of shape (n_components,)
        The amount of variance explained by each of the selected components.
    
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each of the selected components.
    
    components_ : ndarray of shape (n_components, n_features)
        The right singular vectors (V^T).
    
    u_matrix_ : ndarray of shape (n_samples, n_components)
        The left singular vectors (U).
    """
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.components_ = None
        self.u_matrix_ = None
    
    def _validate_data(self, X):
        """Validate input data."""
        X = np.array(X)
        if X.ndim != 2:
            raise ValueError("Expected 2D array, got array with shape %s" % str(X.shape))
        return X
    
    def _get_n_components(self, n_samples, n_features):
        """Determine number of components to keep."""
        if self.n_components is None:
            n_components = min(n_samples, n_features)
        elif isinstance(self.n_components, float):
            if not 0 <= self.n_components <= 1:
                raise ValueError("n_components must be between 0 and 1")
            n_components = min(n_samples, n_features)
        else:
            n_components = min(self.n_components, min(n_samples, n_features))
        return n_components
    
    def fit(self, X):
        """Fit SVD model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_data(X)
        n_samples, n_features = X.shape
        n_components = self._get_n_components(n_samples, n_features)
        
        # Compute SVD
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        
        # Select components
        U = U[:, :n_components]
        S = S[:n_components]
        Vt = Vt[:n_components]
        
        # Calculate explained variance
        explained_variance = (S ** 2) / (n_samples - 1)
        total_var = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_var
        
        # If n_components is float, adjust based on explained variance ratio
        if isinstance(self.n_components, float):
            ratio_cumsum = np.cumsum(explained_variance_ratio)
            n_components = np.sum(ratio_cumsum <= self.n_components) + 1
            U = U[:, :n_components]
            S = S[:n_components]
            Vt = Vt[:n_components]
            explained_variance = explained_variance[:n_components]
            explained_variance_ratio = explained_variance_ratio[:n_components]
        
        # Store results
        self.singular_values_ = S
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance_ratio
        self.components_ = Vt
        self.u_matrix_ = U
        
        return self
    
    def transform(self, X):
        """Apply dimensionality reduction to X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
        
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        X = self._validate_data(X)
        return np.dot(X, self.components_.T)
    
    def fit_transform(self, X):
        """Fit SVD model to X and apply dimensionality reduction to X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        return self.fit(X).u_matrix_ * self.singular_values_
    
    def inverse_transform(self, X):
        """Transform data back to its original space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Data in transformed space.
        
        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Data in original space.
        """
        X = np.array(X)
        return np.dot(X, self.components_)