import unittest

import numpy as np

from pca.pca import PCA

from sklearn.decomposition import PCA as SklearnPCA


class MyTestCase(unittest.TestCase):
    def test_custom_pca_should_be_similar_to_sklearn(self):
        D = np.random.randint(1, 10)
        N = np.random.randint(D, 100)
        K = np.random.randint(1, D)
        X = np.random.rand(N, D)
        print("X shape")
        print(X.shape)
        n_components = K
        print("n_components")
        print(n_components)

        sklearn_pca = SklearnPCA(n_components)
        X_transformed_sklearn = sklearn_pca.fit_transform(X)
        print("sklearn_pca.components_")
        print(sklearn_pca.components_)
        print("=============")

        custom_pca = PCA(n_components)
        X_transformed_custom = custom_pca.fit_transform(X)
        print("custom_pca.components")
        print(custom_pca.components)
        print("=============")

        np.testing.assert_allclose(custom_pca.components, sklearn_pca.components_, rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
