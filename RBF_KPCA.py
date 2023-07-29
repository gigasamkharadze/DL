from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kpca(x, gamma, n):
    """_summary_

    Args:
        x (_Numpy ndarray_ ): _input feature_
        gamma (_float_): _Tuning parameter for rbf kernel_
        n (_int_): _number of components to return_
    """

    # calculate pairwise squared Euclidean distances
    squared_distance = pdist(x, 'sqeuclidean')

    # convert the pairwise distances into a square matrix
    squared_distance_matrix = squareform(squared_distance)

    # compute the symmetric kernel matrix
    kernel_matrix = exp(-gamma * squared_distance_matrix)

    # center the kernel matrix
    n_samples = kernel_matrix.shape[0]
    one_n = np.ones((n_samples, n_samples)) / n_samples
    kernel_matrix = kernel_matrix - one_n.dot(kernel_matrix) - kernel_matrix.dot(one_n) + one_n.dot(kernel_matrix).dot(one_n)

    # obtain eigenpairs from the centered kernel matrix
    eigenvalues, eigenvectors = eigh(kernel_matrix)
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]

    # collect the top k eigenvectors
    x_pc = np.column_stack((eigenvectors[:, i] for i in range(n)))

    return x_pc