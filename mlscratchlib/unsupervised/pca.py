import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) implementation.

    Attributes:
        k (int): The number of components to keep.
        mean (np.ndarray): The mean of the training data features.
        std (np.ndarray): The standard deviation of the training data features.
        components (np.ndarray): The principal components (eigenvectors).
    """

    def __init__(self, n_components):
        """
        Initializes the PCA class.

        Args:
            n_components (int): The number of principal components to keep.
        """
        self.k = n_components
        self.mean = None
        self.std = None
        self.components = None # This will be our W (projection matrix)

    def fit(self, X):
        """
        Fits the PCA model to the data X by computing the principal components.

        Args:
            X (np.ndarray): The dataset to fit (n_samples, n_features).
        """
        # Step 1: Standardize the dataset
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Avoid division by zero if a feature has zero standard deviation
        # A small epsilon is used to prevent RuntimeWarning
        z = (X - self.mean) / (self.std + 1e-8)

        # Step 2: Compute covariance matrix
        # rowvar=False means features are columns
        cov_matrix = np.cov(z, rowvar=False)

        # Step 3: Eigenvector and eigenvalues
        eigenvalues, eigenvector = np.linalg.eig(cov_matrix)

        # Step 4: Sort and select principal component
        # Sort in descending order of eigenvalues
        sorted_ind = np.argsort(eigenvalues)[::-1]

        # Sort eigenvectors based on eigenvalues
        eigenvector = eigenvector[:, sorted_ind]

        # Step 5: Select top k component (Principal Components)
        # Select the first 'k' columns of the sorted eigenvector matrix
        self.components = eigenvector[:, :self.k]

        return self

    def transform(self, X):
        """
        Applies dimensionality reduction to X using the fitted components.

        Args:
            X (np.ndarray): The dataset to transform (n_samples, n_features).

        Returns:
            np.ndarray: The transformed dataset (n_samples, k).
        """
        if self.mean is None or self.std is None or self.components is None:
            raise RuntimeError("The PCA model must be fitted before transforming data.")

        # Standardize the new data using the MEAN and STD from the training data (FIT)
        z = (X - self.mean) / (self.std + 1e-8)

        # Step 6: Transform data (z_pca = z . W)
        z_pca = np.dot(z, self.components)

        return z_pca

