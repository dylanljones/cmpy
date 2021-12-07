# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Tools for Green's functions."""

import numpy as np
import numpy.linalg as la
from mpmath import fp
from functools import partial
from typing import Union

_ellipk_z = np.frompyfunc(partial(fp.ellipf, np.pi / 2), 1, 1)


def gf0_lehmann(
    *args, z: Union[complex, np.ndarray], mu: float = 0.0, mode="diag"
) -> np.ndarray:
    """Calculate the non-interacting Green's function.

    Parameters
    ----------
    *args : tuple of np.ndarray
        Input argument. This can either be a tuple of size two, containing arrays of
        eigenvalues and eigenvectors or a single argument, interpreted as
        Hamilton-operator and used to compute the eigenvalues and eigenvectors used in
        the calculation. The eigenvectors of the Hamiltonian.
    z : (..., Nw) complex np.ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    mu : float, optional
        Chemical potential of the system.
    mode : str, optional
        The output mode of the method. Can either be 'full', 'diag' or 'total'.
        The default is 'diag'. Mode 'full' computes the full Green's function matrix,
        'diag' the diagonal and 'total' computes the trace of the Green's function.

    Returns
    -------
    gf : (...., Nw, N) complex np.ndarray or (...., Nw, N, N) complex np.ndarray
        The Green's function evaluated at `z`.
    """
    if len(args) == 1:
        eigvals, eigvecs = la.eigh(args[0])
    else:
        eigvals, eigvecs = args

    z = np.atleast_1d(z)
    eigvecs_adj = np.conj(eigvecs).T

    if mode == "full":
        subscript_str = "ik,...k,kj->...ij"
    elif mode == "diag":
        subscript_str = "ij,...j,ji->...i"
    elif mode == "total":
        subscript_str = "ij,...j,ji->..."
    else:
        raise ValueError(
            f"Mode '{mode}' not supported. "
            f"Valid modes are 'full', 'diag' or 'total'"
        )
    arg = np.subtract.outer(z + mu, eigvals)
    return np.einsum(subscript_str, eigvecs_adj, 1 / arg, eigvecs)


# =========================================================================
# Common Green's functions
# =========================================================================


def _u_ellipk(z):
    """Complete elliptic integral of first kind `ellip` for complex arguments.

    Wraps the `mpmath` implementation `mpmath.fp.ellipf` using `numpy.frompyfunc`.

    Parameters
    ----------
    z : complex or complex array_like
        Complex argument
    Returns
    -------
    complex np.ndarray or complex
        The complete elliptic integral.
    """
    ellipk = _ellipk_z(np.asarray(z, dtype=complex))
    try:
        ellipk = ellipk.astype(complex)
    except AttributeError:  # complex not np.ndarray
        pass
    return ellipk


def gf_z_bethe(
    z: Union[complex, np.ndarray], half_bandwidth: float
) -> Union[float, np.ndarray]:
    r"""Local Green's function of the Bethe lattice for infinite coordination number.

    Parameters
    ----------
    z : (..., Nw) complex np.ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the 1D lattice.
        Corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    gf : (..., Nw) complex np.ndarray or complex
        The local Green's function.
    """
    dtype = np.complex256
    z_rel = np.array(z / half_bandwidth, dtype=dtype)
    gf = 2.0 / half_bandwidth * z_rel * (1 - np.sqrt(1 - z_rel ** (-2)))
    return gf.astype(dtype=dtype, copy=False)


def gf_z_onedim(
    z: Union[complex, np.ndarray], half_bandwidth: float
) -> Union[float, np.ndarray]:
    """Local Green's function of the 1D lattice.

    Parameters
    ----------
    z : (..., Nw) complex np.ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the 1D lattice.
        Corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    gf : (..., Nw) complex np.ndarray or complex
        The local Green's function.
    """
    return 1.0 / (z * np.lib.scimath.sqrt(1 - (half_bandwidth / z) ** 2))


def gf_z_square(
    z: Union[complex, np.ndarray], half_bandwidth: float
) -> Union[float, np.ndarray]:
    """Local Green's function of the square lattice.

    Parameters
    ----------
    z : (..., Nw) complex np.ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    half_bandwidth : float
        Half-bandwidth of the DOS of the square lattice.
        Corresponds to the nearest neighbor hopping `t=D/2`

    Returns
    -------
    gf : (..., Nw) complex np.ndarray or complex
        The local Green's function.
    """
    z_rel_inv = half_bandwidth / z
    elliptic = _u_ellipk(z_rel_inv ** 2)
    return 2.0 / np.pi / half_bandwidth * z_rel_inv * elliptic
