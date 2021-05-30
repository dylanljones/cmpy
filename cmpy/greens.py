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
from .operators import CreationOperator


def gf0_lehmann(ham, omega, mu=0., only_diag=True):
    """ Calculate the greens function of the given Hamiltonian

    Parameters
    ----------
    ham: array_like
        Hamiltonian matrix
    omega: complex or array_like
        Energy e+eta of the greens function (must be complex!)
    mu: float, default: 0
        Chemical potential of the system
    only_diag: bool, default: True
        only return diagonal elements of the greens function if True

    Returns
    -------
    greens: np.ndarray
    """
    omega = np.atleast_1d(omega)

    # Calculate eigenvalues and -vectors of hamiltonian
    eigvals, eigvecs = np.linalg.eigh(ham)
    eigenvectors_adj = np.conj(eigvecs).T

    # Calculate greens-function
    subscript_str = "ij,...j,ji->...i" if only_diag else "ik,...k,kj->...ij"
    arg = np.subtract.outer(omega + mu, eigvals)
    greens = np.einsum(subscript_str, eigenvectors_adj, 1 / arg, eigvecs)
    return greens


class GreensFunction(np.ndarray):
    """Container class for the single-particle interacting Greens function."""

    def __new__(cls, z, beta, dtype=None):
        r"""Create new `GreensFunction`` instance in the shape of `z`.

        Parameters
        ----------
        z : (N) np.ndarray
            The broadened complex frequency .math:`\omega + i \eta`.
        beta : float
            The inverse temperature .math:`\beta = 1 / T`.
        dtype : str or int or np.dtype, optional
            Optional datatype of the gf-array. The default is the same as `z`.
        """
        inputarr = np.zeros_like(z)
        self = np.asarray(inputarr, dtype).view(cls)
        self.z = z
        self.beta = beta
        return self

    def accumulate(self, cdag, eigvals, eigvecs, eigvals_p1, eigvecs_p1, min_energy=0.):
        """Accumulate the `GreensFunction` iterative.

        Parameters
        ----------
        cdag : CreationOperator
            The creation operator used in the expectation .math:`<n|c^†|m>`.
        eigvals : (N) np.ndarray
            The eigenvalues of the right state .math:`|m>`.
        eigvecs : (N, N) np.ndarray
            The eigenvectors of the right state .math:`|m>`.
        eigvals_p1 : (M) np.ndarray
            The eigenvalues of the left state .math:`<n|`.
        eigvecs_p1 : (M) np.ndarray
            The eigenvectors of the left state .math:`<n|`.
        min_energy : float, optional
            The ground-state energy. The default is ``0``-
        """
        cdag_vec = cdag.matmat(eigvecs)
        overlap = abs(eigvecs_p1.T.conj() @ cdag_vec) ** 2

        if np.isfinite(self.beta):
            exp_eigvals = np.exp(-self.beta * (eigvals - min_energy))
            exp_eigvals_p1 = np.exp(-self.beta * (eigvals_p1 - min_energy))
        else:
            exp_eigvals = np.ones_like(eigvals)
            exp_eigvals_p1 = np.ones_like(eigvals_p1)

        for m, eig_m in enumerate(eigvals_p1):
            for n, eig_n in enumerate(eigvals):
                weights = exp_eigvals[n] + exp_eigvals_p1[m]
                self[:] += overlap[m, n] * weights / (self.z + eig_n - eig_m)
