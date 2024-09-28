import numpy as np

from pca.utils import get_covariance_matrix, get_eigenvalues_and_eigenvectors


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.tolerance = 1e-6
        self.max_iter = 1000
        self.mean = None

    def fit(self, X):
        # Step 1: Shift the data to have zero mean
        self.mean = np.mean(X, axis=0) # column-wise
        X = X - self.mean

        # Step 2: Compute the covariance matrix
        cov_maxtrix = get_covariance_matrix(X) # Positive semi-definite matrix

        # Step 3: Compute the eigenvalues and eigenvectors of the covariance matrix
        eigen_values, eigen_vectors = get_eigenvalues_and_eigenvectors(cov_maxtrix, self.n_components)

        # Step 4: Sort the eigenvectors based on the eigenvalues
        sorted_indices = np.argsort(eigen_values)[::-1]
        eigen_vectors = eigen_vectors[sorted_indices]

        # Step 5: Select the first n_components eigenvectors
        self.components = eigen_vectors[:self.n_components]

    def transform(self, X):
        # Step 1: Shift the data to have zero mean (for inference)
        X = X - self.mean

        # Step 2: Project the data onto the new subspace
        return np.dot(X, self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)