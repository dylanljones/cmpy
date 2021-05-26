"""Estimate for 1-norm.

References
----------
http://eprints.maths.manchester.ac.uk/321/1/35608.pdf

Other references:
https://www.maths.manchester.ac.uk/~higham/narep/narep176.pdf
https://link.springer.com/content/pdf/10.1007/BF01396242.pdf

"""

import warnings

import numpy as np

from numpy.random import default_rng
from scipy.sparse.linalg import LinearOperator

RNG = default_rng()


def sign(x):
    if not np.iscomplexobj(x):
        return np.where(x >= 0, np.int8(1), np.int8(-1))
    with warnings.catch_warnings():  # ignore warning for 0/abs(0)
        warnings.filterwarnings(action='ignore', category=RuntimeWarning,
                                message=".*in true_divide.*")
        return np.where(x == 0, 1.0, x / abs(x))


def estimate_1norm(A: LinearOperator, t: int):
    """Estimation of the norm of the linear operator `A`.

    Implements:
    http://eprints.maths.manchester.ac.uk/321/1/35608.pdf

    I have just seen that there is also a implementation hidden in scipy.

    """
    assert t >= 1
    n, m = A.shape
    assert n == m, "Works only for square operators."
    del m
    MAX_ITER = 100

    # normalized random starting point
    X = RNG.random([n, t]) - 0.5
    norm = np.linalg.norm(X, ord=1, axis=0)
    X /= norm

    est_old = 0
    ind_hist = []
    ind = np.zeros(n, dtype=int)
    S = np.zeros([n, t], dtype=np.int8)
    for k in range(1, MAX_ITER):
        Y = A.matmat(X)  # matmat faster than A @ X
        norms = np.linalg.norm(Y, ord=1, axis=0)
        est = max(norms)

        if est > est_old or k == 2:
            ind_best = np.argmax(norms)
        if k >= 2 and est <= est_old:
            return est_old

        est_old, Sold = est, S
        S = sign(Y)
        # we skip the check, if every column of S is parallel to a column in Sold
        if t > 1:  # find parallel vectors, I don't think this works for complex...
            parallel = abs(np.einsum('ij,ik->jk', S, S)) / t**2
            for j, k in np.argwhere(parallel > 1 - 1e-6):
                if j > k:
                    S[:, j] = RNG.choice([-1, +1], size=n)
            parallel = abs(np.einsum('ij,ik->jk', S, Sold)) / t**2
            for j, k in np.argwhere(parallel > 1 - 1e-6):
                S[:, j] = RNG.choice([-1, +1], size=n)
        Z = A.adjoint().matmat(S)
        h = np.linalg.norm(Z, ord=np.infty, axis=1)
        if k >= 2 and max(h) == h[ind_best]:
            return est
        ind = np.argsort(h)[::-1]
        h = h[ind]
        if t > 1:
            if np.setdiff1d(ind[:t], ind_hist, assume_unique=True).size == 0:
                return est
            ind = ind[~np.isin(ind, ind_hist)][:t]
        X = np.zeros([n, t])
        for ti in range(t):
            X[ind[ti], ti] = 1
        ind_hist.extend(ind)
    print("Maximum iteration reached")
    return est
