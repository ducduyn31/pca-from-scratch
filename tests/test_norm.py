import unittest

import numpy as np

from pca.utils import get_matrix_norm


class MyTestCase(unittest.TestCase):
    def test_custom_norm_should_match_norm(self):
        X = np.random.rand(30, 6)
        print("X")
        print(X)
        print("=============")
        print("np.linalg.norm")
        norm = np.linalg.norm(X)
        print(norm)
        print("=============")
        print("get_matrix_normalization")
        custom_norm = get_matrix_norm(X)
        print(custom_norm)

        self.assertAlmostEqual(norm, custom_norm)

if __name__ == '__main__':
    unittest.main()
