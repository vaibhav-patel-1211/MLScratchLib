import numpy as np
from scipy.spatial.distance import cdist

class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
    
    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
        
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered
        as a core point.
        
    metric : str, default='euclidean'
        The metric to use when calculating distance between instances.
        Supported metrics are those in scipy.spatial.distance.
    
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point in the dataset.
        Noisy samples are given the label -1.
        
    core_sample_indices_ : ndarray of shape (n_core_samples,)
        Indices of core samples.
    """
    
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None
        self.core_sample_indices_ = None
    
    def _find_neighbors(self, X, sample_idx):
        """Find all points within eps distance of sample_idx."""
        distances = cdist(X[sample_idx:sample_idx+1], X, metric=self.metric)[0]
        return np.where(distances <= self.eps)[0]
    
    def fit(self, X):
        """Perform DBSCAN clustering.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Initialize labels as noise
        self.labels_ = np.full(n_samples, -1)
        
        # Find core points
        core_samples = []
        for sample_idx in range(n_samples):
            neighbors = self._find_neighbors(X, sample_idx)
            if len(neighbors) >= self.min_samples:
                core_samples.append(sample_idx)
        
        self.core_sample_indices_ = np.array(core_samples)
        
        # Assign cluster labels
        current_label = 0
        for core_idx in core_samples:
            if self.labels_[core_idx] != -1:
                continue
                
            # Start a new cluster
            self.labels_[core_idx] = current_label
            
            # Find all points density-reachable from current core point
            stack = [core_idx]
            while stack:
                current_point = stack.pop()
                neighbors = self._find_neighbors(X, current_point)
                
                for neighbor_idx in neighbors:
                    if self.labels_[neighbor_idx] == -1:
                        self.labels_[neighbor_idx] = current_label
                        
                        # If neighbor is core point, add to stack
                        if neighbor_idx in core_samples:
                            stack.append(neighbor_idx)
            
            current_label += 1
        
        return self
    
    def fit_predict(self, X):
        """Perform DBSCAN clustering and return cluster labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels. Noisy samples are given the label -1.
        """
        return self.fit(X).labels_