import numpy as np


def get_covariance_matrix(X):
    """
    Compute the covariance matrix of the input matrix X based on the formula:
    cov(X) = 1/N * X * X^T
    :param X: input matrix
    :return: covariance matrix
    """
    N = X.shape[0] - 1
    X_bar = X - np.mean(X, axis=0)
    return np.dot(X_bar.T, X_bar) / N

def get_eigenvalues_and_eigenvectors(cov_matrix: np.ndarray, n_components: int):
    """
    Compute the eigenvalues and eigenvectors of the input covariance matrix
    :param cov_matrix: covariance matrix
    :param n_components: number of components
    :return: eigenvalues, eigenvectors
    """
    eigen_values = []
    eigen_vectors = []

    matrix_copy = cov_matrix.copy()
    for _ in range(n_components):
        eigenvalue, eigenvector = power_iteration(matrix_copy)
        eigen_values.append(eigenvalue)
        eigen_vectors.append(eigenvector)

        matrix_copy -= eigenvalue * np.outer(eigenvector, eigenvector)

    return np.array(eigen_values), np.array(eigen_vectors)

def power_iteration(matrix: np.ndarray, max_iter: int = 1e3, tolerance: float = 1e-6):
    """
    Compute the dominant eigenvector of the input matrix using the power iteration method
    :param matrix: input matrix
    :param max_iter: maximum number of iterations
    :param tolerance: tolerance
    :return: dominant eigenvector
    """
    n = matrix.shape[0]

    # Step 1: Select a random vector q with ||q|| = 1
    q_k = np.random.rand(n)

    for _ in range(int(max_iter)):
        # Step 2: Compute the vector z = A * q_k
        z = np.dot(matrix, q_k) # MxN * Nx1 = Mx1

        # Step 3:  Normalize z to get the next vector q_{k+1}
        z_norm = get_matrix_norm(z)
        q_next = z / z_norm

        # Step 4: Check for convergence
        if get_matrix_norm(q_next - q_k) < tolerance:
            break

        q_k = q_next

    # Step 5: Compute the eigenvalue, the eigenvector is the final q_k
    eigenvalue = np.dot(q_k.T, np.dot(matrix, q_k))
    eigenvector = q_k

    return eigenvalue, eigenvector

def get_matrix_norm(X):
    """
    Compute the L2 norm of the input matrix
    :param X: input matrix
    :return: norm
    """
    return np.sqrt(np.sum(X ** 2))

def find_optimal_k(X, threshold=0.95):
    """
    Find the optimal number of components k that explain at least threshold of the variance
    :param X: input matrix
    :param threshold: variance threshold
    :return: optimal number of components
    """
    # Step 1: Compute the covariance matrix
    cov_matrix = get_covariance_matrix(X)

    # Step 2: Compute the eigenvalues
    eigen_values, _ = get_eigenvalues_and_eigenvectors(cov_matrix, X.shape[1])
    eigen_values = np.sort(eigen_values)[::-1]

    # Step 3: Compute the optimal number of components
    total_variance = np.sum(eigen_values)
    explained_variance = np.cumsum(eigen_values) / total_variance
    k = np.argmax(explained_variance >= threshold) + 1

    return k
