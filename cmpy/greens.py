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
from .basis import UP
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


def accumulate_gf(gf, z, beta, cdag_vec, eigvals, eigvals_p1, eigvecs_p1):
    overlap = abs(eigvecs_p1.T.conj() @ cdag_vec) ** 2
    exp_eigvals = np.exp(-beta * eigvals)
    exp_eigvals_p1 = np.exp(-beta * eigvals_p1)
    for m, eig_m in enumerate(eigvals_p1):
        for n, eig_n in enumerate(eigvals):
            weights = exp_eigvals[n] + exp_eigvals_p1[m]
            gf += overlap[m, n] * weights / (z + eig_n - eig_m)


def impurity_gf(siam, z, beta=1., sigma=UP):
    partition = 0
    gf_imp = np.zeros_like(z)

    for n_up, n_dn in siam.iter_sector_keys(2):

        # Solve current particle sector
        sector = siam.get_sector(n_up, n_dn)
        hamop = siam.hamiltonian_op(sector=sector)
        ham = hamop.matmat(np.eye(*hamop.shape))
        eigvals, eigvecs = la.eigh(ham)

        # Accumulate partition
        partition += np.sum(np.exp(-beta * eigvals))

        # Accumulate Green's function
        if (n_up + 1) in siam.sector_keys:
            sector_p1 = siam.get_sector(n_up + 1, n_dn)
            hamop = siam.hamiltonian_op(sector=sector_p1)
            ham = hamop.matmat(np.eye(*hamop.shape))
            eigvals_p1, eigvecs_p1 = la.eigh(ham)

            cdag = CreationOperator(sector, sector_p1, pos=0, sigma=sigma)
            cdag_vec = cdag.matmat(eigvecs)

            accumulate_gf(gf_imp, z, beta, cdag_vec, eigvals, eigvals_p1, eigvecs_p1)

    return gf_imp / partition
