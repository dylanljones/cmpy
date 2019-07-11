# -*- coding: utf-8 -*-
"""
Created on 14 Feb 2019
author: Dylan

project: cmpy
version: 1.0
"""
import numpy as np
from scipy import linalg as la
from sciutils import eig_banded


def gf(ham, omega, expand=True):
    """ Calculate the Green's function for a given non-interacting hamiltonian

    Parameters
    ----------
    omega: float
        energy value to calculate the Green's function
    ham: array_like
        hamiltonian
    expand: bool, default: True
        Flag if energy has to be converted to matrix form

    Returns
    -------
    gf: array_like
    """
    if expand:
        omega = np.eye(ham.shape[0]) * omega
    return la.inv(omega - ham, overwrite_a=True, check_finite=False)


def advanced_gf(omega, ham, sigma_l, sigma_r):
    """ Calculate the advanced Green's function of a device between leads

    Parameters
    ----------
    omega: float
        energy value to calculate the advanced gf
    ham: array_like
        hamiltonian of the device
    sigma_l: array_like
        self-energy of the left lead
    sigma_r: array_like
        self-energy of the right lead
    Returns
    -------
    gf: array_like
    """
    return la.inv(omega * np.eye(ham.shape[0]) - ham - sigma_l - sigma_r)


def gf_lehmann(ham, omega, mu=0., only_diag=True, banded=False):
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
    banded: bool, default: False
        Use the upper diagonal matrix for solving the eigenvalue problem.
        Uses full diagonalization as default

    Returns
    -------
    greens: np.ndarray
    """
    scalar = False
    if not hasattr(omega, "__len__"):
        scalar = True
        omega = [omega]
    omega = np.asarray(omega)

    # Calculate eigenvalues and -vectors of hamiltonian
    if banded:
        eigvals, eigvecs = eig_banded(ham)
    else:
        eigvals, eigvecs = np.linalg.eig(ham)
    eigenvectors_adj = np.conj(eigvecs).T

    # Calculate greens-function
    subscript_str = "ij,...j,ji->...i" if only_diag else "ik,...k,kj->...ij"
    arg = np.subtract.outer(omega + mu, eigvals)
    greens = np.einsum(subscript_str, eigenvectors_adj, 1 / arg, eigvecs)
    return greens if not scalar else greens[0]


def rda(ham, t, omega, thresh=0.):
    """ Calculate the left, bulk and right Green's function of a (half-)infinite system

    Calculate Calculate the left, bulk and right Green's function using the recursive decimation
    algorithm

    Parameters
    ----------
    ham: array_like
        hamiltonian of one slice of the system
    t: array_like
        coupling matrix between the slices
    omega: float
        energy-value to calculate the Green's functions
    thresh: float, optional
        threshold to consider decimation convergent, default 0.

    Returns
    -------
    gf_l: array_like
    gf_c: array_like
    gf_r: array_like
    """
    omega = omega * np.eye(ham.shape[0])

    alpha = t
    beta = np.conj(t).T
    h_l = ham.copy()
    h_b = ham.copy()
    h_r = ham.copy()
    while True:
        # calculate greens-functions
        gf_l = gf(h_l, omega, expand=False)
        gf_b = gf(h_b, omega, expand=False)
        gf_r = gf(h_l, omega, expand=False)

        # Recalculate hamiltonians
        h_l = h_l + alpha @ gf_b @ beta
        h_b = h_b + alpha @ gf_b @ beta + beta @ gf_b @ alpha
        h_r = h_r + beta @ gf_b @ alpha

        # Check if converged
        val = np.linalg.norm(alpha) + np.linalg.norm(beta)
        if val <= thresh:
            break

        # Renormalize effective hopping
        alpha = alpha @ gf_b @ alpha
        beta = beta @ gf_b @ beta

    return gf_l, gf_b, gf_r
