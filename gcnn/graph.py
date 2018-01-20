import numpy as np
import scipy as sp
import scipy.sparse, scipy.linalg


def grid(n: int):
    """
    Coordinates (x, y) list for a grid of size n x n.
    """
    idx = np.arange(n)
    return np.reshape(np.meshgrid(idx, idx), (2, -1)).T


def fourier(laplacian):
    """
    Graph fourier basis for a laplacian using SVD.
    """
    return sp.linalg.svd(laplacian)[0]

