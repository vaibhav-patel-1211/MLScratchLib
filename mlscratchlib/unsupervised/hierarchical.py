import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from ..utils.visualization import plot_dendrogram

class HierarchicalClustering:
    """Hierarchical Clustering implementation supporting both agglomerative and divisive methods.
    
    Parameters
    ----------
    n_clusters : int, default=2
        The number of clusters to find.
    
    method : {'divisive', 'agglomerative'}, default='agglomerative'
        The type of hierarchical clustering to perform.
        
    linkage : {'single', 'complete', 'average'}, default='complete'
        The linkage criterion to use for agglomerative clustering:
        - 'single': uses minimum distance between clusters
        - 'complete': uses maximum distance between clusters
        - 'average': uses average distance between clusters
    
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point.
        
    n_leaves_ : int
        Number of leaves in the hierarchical tree.
        
    children_ : ndarray of shape (n_nodes-1, 2)
        The children of each non-leaf node. Values less than n_samples refer
        to leaves of the tree. For agglomerative clustering, a bigger value i
        indicates node with index i + n_samples formed by merging its children.
    """
    
    def __init__(self, n_clusters=2, method='agglomerative', linkage='complete'):
        self.n_clusters = n_clusters
        self.method = method
        self.linkage = linkage
        self.labels_ = None
        self.n_leaves_ = None
        self.children_ = None
    
    def _compute_distance_matrix(self, X):
        """Compute the distance matrix for the input data."""
        return squareform(pdist(X))
    
    def _get_cluster_distance(self, dist_matrix, cluster1, cluster2):
        """Compute distance between two clusters based on linkage criterion."""
        distances = dist_matrix[np.ix_(cluster1, cluster2)]
        
        if self.linkage == 'single':
            return np.min(distances)
        elif self.linkage == 'complete':
            return np.max(distances)
        else:  # average
            return np.mean(distances)
    
    def _agglomerative_clustering(self, X):
        """Perform agglomerative hierarchical clustering."""
        n_samples = X.shape[0]
        self.n_leaves_ = n_samples
        
        # Initialize each point as a cluster
        current_clusters = [[i] for i in range(n_samples)]
        labels = np.arange(n_samples)
        children = []
        
        # Compute initial distance matrix
        dist_matrix = self._compute_distance_matrix(X)
        
        while len(current_clusters) > self.n_clusters:
            min_dist = float('inf')
            merge_clusters = (0, 1)
            
            # Find closest clusters to merge
            for i in range(len(current_clusters)):
                for j in range(i + 1, len(current_clusters)):
                    dist = self._get_cluster_distance(
                        dist_matrix,
                        current_clusters[i],
                        current_clusters[j]
                    )
                    if dist < min_dist:
                        min_dist = dist
                        merge_clusters = (i, j)
            
            # Merge clusters
            i, j = merge_clusters
            new_cluster = current_clusters[i] + current_clusters[j]
            new_label = n_samples + len(children)
            
            # Update labels
            for idx in new_cluster:
                labels[idx] = new_label
            
            # Update children and clusters
            children.append([min(current_clusters[i][0], current_clusters[j][0]),
                           max(current_clusters[i][0], current_clusters[j][0])])
            current_clusters = ([new_cluster] +
                              current_clusters[:i] +
                              current_clusters[i+1:j] +
                              current_clusters[j+1:])
        
        self.children_ = np.array(children)
        self.labels_ = labels
        return self
    
    def _divisive_clustering(self, X):
        """Perform divisive hierarchical clustering using k-means for splitting."""
        n_samples = X.shape[0]
        self.n_leaves_ = n_samples
        
        # Initialize with all points in one cluster
        labels = np.zeros(n_samples, dtype=int)
        children = []
        clusters_to_split = [(0, np.arange(n_samples))]
        next_label = 1
        
        while len(clusters_to_split) > 0 and next_label < self.n_clusters:
            parent_label, cluster_indices = clusters_to_split.pop(0)
            
            if len(cluster_indices) <= 1:
                continue
            
            # Split cluster using k-means
            kmeans = KMeans(n_clusters=2, n_init=10)
            sub_labels = kmeans.fit_predict(X[cluster_indices])
            
            # Create new clusters
            left_indices = cluster_indices[sub_labels == 0]
            right_indices = cluster_indices[sub_labels == 1]
            
            # Update labels
            labels[left_indices] = next_label
            labels[right_indices] = next_label + 1
            
            # Add children information
            children.append([parent_label, next_label])
            children.append([parent_label, next_label + 1])
            
            # Add new clusters to split queue
            if next_label + 2 < self.n_clusters:
                clusters_to_split.extend([
                    (next_label, left_indices),
                    (next_label + 1, right_indices)
                ])
            
            next_label += 2
        
        self.children_ = np.array(children)
        self.labels_ = labels
        return self
    
    def fit(self, X):
        """Fit the hierarchical clustering on the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = np.array(X)
        
        if self.method == 'agglomerative':
            return self._agglomerative_clustering(X)
        else:
            return self._divisive_clustering(X)
    
    def fit_predict(self, X):
        """Fit the hierarchical clustering and return cluster labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        return self.fit(X).labels_