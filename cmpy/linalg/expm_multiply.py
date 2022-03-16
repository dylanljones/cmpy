# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2022, Dylan Jones

import numpy as np
from scipy.sparse.linalg import onenormest
from scipy.sparse.linalg.interface import IdentityOperator, LinearOperator  # noqa
from scipy.sparse.linalg._expm_multiply import (
    _expm_multiply_interval_core_0,
    _expm_multiply_interval_core_1,
    _expm_multiply_interval_core_2,
    _expm_multiply_simple_core,
    _fragment_3_1,
    LazyOperatorNormInfo,
)


def _trace(a, **kwargs):
    """If `np.trace` doesn't work hope `a` implements a trace."""
    try:
        return np.trace(a, **kwargs)
    except ValueError:
        return a.trace()


def _identity_like(a):
    if isinstance(a, LinearOperator):
        return IdentityOperator(shape=a.shape)
    return np.eye(*a.shape)


# noinspection PyPep8Naming
def _expm_multiply_interval(
    A,
    B,
    start=None,
    stop=None,
    num=None,
    endpoint=None,
    balance=False,
    status_only=False,
):
    """Compute the action of the matrix exponential at multiple time points.

    Parameters
    ----------
    A : transposable linear operator
        The operator whose exponential is of interest.
    B : ndarray
        The matrix to be multiplied by the matrix exponential of A.
    start : scalar, optional
        The starting time point of the sequence.
    stop : scalar, optional
        The end time point of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced time points, so that `stop` is excluded.
        Note that the step size changes when `endpoint` is False.
    num : int, optional
        Number of time points to use.
    endpoint : bool, optional
        If True, `stop` is the last time point. Otherwise, it is not included.
    balance : bool
        Indicates whether or not to apply balancing.
    status_only : bool
        A flag that is set to True for some debugging and testing operations.

    Returns
    -------
    F : ndarray
        :math:`e^{t_k A} B`
    status : int
        An integer status for testing and debugging.

    Notes
    -----
    This is algorithm (5.2) in Al-Mohy and Higham (2011).
    There seems to be a typo, where line 15 of the algorithm should be
    moved to line 6.5 (between lines 6 and 7).
    """
    if balance:
        raise NotImplementedError
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expected A to be like a square matrix")
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            "shapes of matrices A {} and B {} are incompatible".format(A.shape, B.shape)
        )
    ident = _identity_like(A)
    n = A.shape[0]
    if len(B.shape) == 1:
        n0 = 1
    elif len(B.shape) == 2:
        n0 = B.shape[1]
    else:
        raise ValueError("expected B to be like a matrix or a vector")
    u_d = 2**-53
    tol = u_d
    mu = _trace(A) / float(n)

    # Get the linspace samples, attempting to preserve the linspace defaults.
    linspace_kwargs = {"retstep": True}
    if num is not None:
        linspace_kwargs["num"] = num
    if endpoint is not None:
        linspace_kwargs["endpoint"] = endpoint
    samples, step = np.linspace(start, stop, **linspace_kwargs)

    # Convert the linspace output to the notation used by the publication.
    nsamples = len(samples)
    if nsamples < 2:
        raise ValueError("at least two time points are required")
    q = nsamples - 1
    h = step
    t_0 = samples[0]
    t_q = samples[q]

    # Define the output ndarray.
    # Use an ndim=3 shape, such that the last two indices
    # are the ones that may be involved in level 3 BLAS operations.
    X_shape = (nsamples,) + B.shape
    X = np.empty(X_shape, dtype=np.result_type(A.dtype, B.dtype, float))
    t = t_q - t_0
    A = A - mu * ident
    A_1_norm = onenormest(A, t=10)
    ell = 2
    norm_info = LazyOperatorNormInfo(t * A, A_1_norm=t * A_1_norm, ell=ell)
    if t * A_1_norm == 0:
        m_star, s = 0, 1
    else:
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)

    # Compute the expm action up to the initial time point.
    X[0] = _expm_multiply_simple_core(A, B, t_0, mu, m_star, s)

    # Compute the expm action at the rest of the time points.
    if q <= s:
        if status_only:
            return 0
        else:
            return _expm_multiply_interval_core_0(
                A, X, h, mu, q, norm_info, tol, ell, n0
            )
    elif not (q % s):
        if status_only:
            return 1
        else:
            return _expm_multiply_interval_core_1(A, X, h, mu, m_star, s, q, tol)
    elif q % s:
        if status_only:
            return 2
        else:
            return _expm_multiply_interval_core_2(A, X, h, mu, m_star, s, q, tol)
    else:
        raise Exception("internal error")


# noinspection PyPep8Naming
def _expm_multiply_simple(A, B, t=1.0, balance=False):
    """
    Compute the action of the matrix exponential at a single time point.

    Parameters
    ----------
    A : transposable linear operator
        The operator whose exponential is of interest.
    B : ndarray
        The matrix to be multiplied by the matrix exponential of A.
    t : float
        A time point.
    balance : bool
        Indicates whether or not to apply balancing.

    Returns
    -------
    F : ndarray
        :math:`e^{t A} B`

    Notes
    -----
    This is algorithm (3.2) in Al-Mohy and Higham (2011).
    """
    if balance:
        raise NotImplementedError
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expected A to be like a square matrix")
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            "shapes of matrices A {} and B {} are incompatible".format(A.shape, B.shape)
        )
    ident = _identity_like(A)
    n = A.shape[0]
    if len(B.shape) == 1:
        n0 = 1
    elif len(B.shape) == 2:
        n0 = B.shape[1]
    else:
        raise ValueError("expected B to be like a matrix or a vector")
    u_d = 2**-53
    tol = u_d
    mu = _trace(A) / float(n)
    A = A - mu * ident
    A_1_norm = onenormest(A, t=10)
    if t * A_1_norm == 0:
        m_star, s = 0, 1
    else:
        ell = 2
        norm_info = LazyOperatorNormInfo(t * A, A_1_norm=t * A_1_norm, ell=ell)
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)
    return _expm_multiply_simple_core(A, B, t, mu, m_star, s, tol, balance)


# noinspection PyPep8Naming
def expm_multiply(A, B, start=None, stop=None, num=None, endpoint=None):
    """
    Compute the action of the matrix exponential of A on B.
    Parameters
    ----------
    A : transposable linear operator
        The operator whose exponential is of interest.
    B : ndarray
        The matrix or vector to be multiplied by the matrix exponential of A.
    start : scalar, optional
        The starting time point of the sequence.
    stop : scalar, optional
        The end time point of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced time points, so that `stop` is excluded.
        Note that the step size changes when `endpoint` is False.
    num : int, optional
        Number of time points to use.
    endpoint : bool, optional
        If True, `stop` is the last time point.  Otherwise, it is not included.
    Returns
    -------
    expm_A_B : ndarray
         The result of the action :math:`e^{t_k A} B`.
    Notes
    -----
    The optional arguments defining the sequence of evenly spaced time points
    are compatible with the arguments of `numpy.linspace`.
    The output ndarray shape is somewhat complicated so I explain it here.
    The ndim of the output could be either 1, 2, or 3.
    It would be 1 if you are computing the expm action on a single vector
    at a single time point.
    It would be 2 if you are computing the expm action on a vector
    at multiple time points, or if you are computing the expm action
    on a matrix at a single time point.
    It would be 3 if you want the action on a matrix with multiple
    columns at multiple time points.
    If multiple time points are requested, expm_A_B[0] will always
    be the action of the expm at the first time point,
    regardless of whether the action is on a vector or a matrix.
    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2011)
           "Computing the Action of the Matrix Exponential,
           with an Application to Exponential Integrators."
           SIAM Journal on Scientific Computing,
           33 (2). pp. 488-511. ISSN 1064-8275
           http://eprints.ma.man.ac.uk/1591/
    .. [2] Nicholas J. Higham and Awad H. Al-Mohy (2010)
           "Computing Matrix Functions."
           Acta Numerica,
           19. 159-208. ISSN 0962-4929
           http://eprints.ma.man.ac.uk/1451/
    Examples
    --------
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import expm, expm_multiply
    >>> A = csc_matrix([[1, 0], [0, 1]])
    >>> A.todense()
    matrix([[1, 0],
            [0, 1]], dtype=int64)

    >>> B = np.array([np.exp(-1.), np.exp(-2.)])
    >>> B
    array([ 0.36787944,  0.13533528])

    >>> expm_multiply(A, B, start=1, stop=2, num=3, endpoint=True)
    array([[ 1.        ,  0.36787944],
           [ 1.64872127,  0.60653066],
           [ 2.71828183,  1.        ]])

    >>> expm(A).dot(B)                  # Verify 1st timestep
    array([ 1.        ,  0.36787944])

    >>> expm(1.5*A).dot(B)              # Verify 2nd timestep
    array([ 1.64872127,  0.60653066])

    >>> expm(2*A).dot(B)                # Verify 3rd timestep
    array([ 2.71828183,  1.        ])
    """
    if all(arg is None for arg in (start, stop, num, endpoint)):
        X = _expm_multiply_simple(A, B)
    else:
        X, status = _expm_multiply_interval(A, B, start, stop, num, endpoint)
    return X
