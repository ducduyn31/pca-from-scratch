import unittest
import numpy as np

from pca.utils import get_covariance_matrix


class MyTestCase(unittest.TestCase):
    def test_custom_covariance_matrix_should_be_similar_to_numpy(self):
        X = np.random.rand(30, 6)
        print("X")
        print(X)
        print("=============")
        print("np.cov")
        print(np.cov(X, rowvar=False))
        print("=============")
        print("get_covariance_matrix")
        print(get_covariance_matrix(X))
        self.assertTrue(np.allclose(get_covariance_matrix(X), np.cov(X, rowvar=False)))


if __name__ == '__main__':
    unittest.main()
