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

_ellipk_z = np.frompyfunc(partial(fp.ellipf, np.pi/2), 1, 1)


def gf0_lehmann(ham: np.ndarray, z: Union[complex, np.ndarray],
                mu: float = 0., only_diag: bool = True) -> np.ndarray:
    """Calculate the non-interacting Green's function.

    Parameters
    ----------
    ham : (N, N) np.ndarray
        Array representing the Hamiltonian matrix.
    z : (..., Nw) complex np.ndarray or complex
        Green's function is evaluated at complex frequency `z`.
    mu : float, optional
        Chemical potential of the system.
    only_diag : bool, optional
        Only return diagonal elements of the greens function if `True`.

    """
    z = np.atleast_1d(z)

    # Calculate eigenvalues and -vectors of hamiltonian
    eigvals, eigvecs = np.linalg.eigh(ham)
    eigenvectors_adj = np.conj(eigvecs).T

    # Calculate greens-function
    subscript_str = "ij,...j,ji->...i" if only_diag else "ik,...k,kj->...ij"
    arg = np.subtract.outer(z + mu, eigvals)
    greens = np.einsum(subscript_str, eigenvectors_adj, 1 / arg, eigvecs)
    return greens


# =========================================================================
# Common Green's functions
# =========================================================================


def _u_ellipk(z):
    """Complete elliptic integral of first kind `scipy.special.ellip` for complex arguments.

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


def gf_z_bethe(z: Union[complex, np.ndarray], half_bandwidth: float) -> Union[float, np.ndarray]:
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
    gf = 2. / half_bandwidth * z_rel * (1 - np.sqrt(1 - z_rel**(-2)))
    return gf.astype(dtype=dtype, copy=False)


def gf_z_onedim(z: Union[complex, np.ndarray], half_bandwidth: float) -> Union[float, np.ndarray]:
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
    return 1. / (z * np.lib.scimath.sqrt(1 - (half_bandwidth / z) ** 2))


def gf_z_square(z: Union[complex, np.ndarray], half_bandwidth: float) -> Union[float, np.ndarray]:
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
    z_rel_inv = half_bandwidth/z
    elliptic = _u_ellipk(z_rel_inv**2)
    return 2./np.pi/half_bandwidth*z_rel_inv*elliptic
