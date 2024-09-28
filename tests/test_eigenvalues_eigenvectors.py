import unittest

import numpy as np

from pca.utils import get_eigenvalues_and_eigenvectors


class MyTestCase(unittest.TestCase):
    def test_eigen_values_eigen_vectors_calculations(self):
        N = np.random.randint(1, 100)
        D = np.random.randint(1, 50)
        X = np.random.rand(N, D)
        cov = np.cov(X, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eig(cov)
        custom_eigen_values, custom_eigen_vectors = get_eigenvalues_and_eigenvectors(cov, X.shape[1])

        # Sort the eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[sorted_indices]
        eigen_vectors = eigen_vectors[:, sorted_indices]

        # Sort the custom eigenvalues and eigenvectors
        sorted_indices = np.argsort(custom_eigen_values)[::-1]
        custom_eigen_values = custom_eigen_values[sorted_indices]
        custom_eigen_vectors = custom_eigen_vectors[:, sorted_indices]

        print("eigen_values")
        print(eigen_values)
        print("=============")
        print("custom_eigen_values")
        print(custom_eigen_values)
        print("=============")
        print("eigen_vectors")
        print(eigen_vectors)
        print("=============")
        print("custom_eigen_vectors")
        print(custom_eigen_vectors)

        self.assertTrue(np.allclose(eigen_values, custom_eigen_values))


if __name__ == '__main__':
    unittest.main()
