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


def freq_tail_fourier(tail_coef, beta, tau, w_n):
    r"""Fourier transforms analytically the slow decaying tail_coefs of
    the Greens functions [matsubara]

    See also
    --------
    gf_fft

    References
    ----------
    [matsubara] https://en.wikipedia.org/wiki/Matsubara_frequency#Time_Domain
    """
    freq_tail = tail_coef[0] / (1.j * w_n) + tail_coef[1] / (1.j * w_n)**2 + tail_coef[2] / (1.j * w_n)**3
    time_tail = - tail_coef[0] / 2 + tail_coef[1] / 2 * (tau - beta / 2) - tail_coef[2] / 4 * (tau**2 - beta * tau)
    return freq_tail, time_tail


def gf_tau_fft(gf_tau, tau, omega, tail_coef=(1., 0., 0.)):
    """ Perform a fourier transform on the imaginary time Green's function

    Parameters
    ----------
    gf_tau : real float array
        Imaginary time green's function to transform
    tau : real float array
        Imaginary time points
    omega : real float array
        fermionic matsubara frequencies. Only use the positive ones
    tail_coef : list of floats size 3
        The first moments of the tails

    Returns
    -------
    gf_omega: complex ndarray or complex
        Value of the transformed imaginary frequency green's function
    """
    beta = tau[1] + tau[-1]
    freq_tail, time_tail = freq_tail_fourier(tail_coef, beta, tau, omega)

    gf_tau = gf_tau - time_tail
    gf_omega = beta * np.fft.ifft(gf_tau * np.exp(1j * np.pi * tau / beta))[..., :len(omega)] + freq_tail
    return gf_omega


def gf_omega_fft(gf_omega, tau, omega, tail_coef=(1., 0., 0.)):
    """ Perform a fourier transform on the imaginary frequency Green's function

    Parameters
    ----------
    gf_omega : real float array
        Imaginary frequency Green's function to transform
    tau : real float array
        Imaginary time points
    omega : real float array
        fermionic matsubara frequencies. Only use the positive ones
    tail_coef : list of floats size 3
        The first moments of the tails

    Returns
    -------
    gf_tau: complex ndarray or complex
        Value of the transformed imaginary time green's function
    """
    beta = tau[1] + tau[-1]
    freq_tail, time_tail = freq_tail_fourier(tail_coef, beta, tau, omega)

    gf_omega = gf_omega - freq_tail
    gf_tau = np.fft.fft(gf_omega, len(tau)) * np.exp(-1j * np.pi * tau / beta)
    gf_tau = (2 * gf_tau / beta).real + time_tail
    return gf_tau


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


def bethe_gf_omega(z, half_bandwidth=1):
    """Local Green's function of Bethe lattice for infinite Coordination number.

    Taken from gf_tools by Weh Andreas
    https://github.com/DerWeh/gftools/blob/master/gftools/__init__.py

    Parameters
    ----------
    z : complex ndarray or complex
        Green's function is evaluated at complex frequency `z`
    half_bandwidth : float
        half-bandwidth of the DOS of the Bethe lattice
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`
    Returns
    -------
    bethe_gf_omega : complex ndarray or complex
        Value of the Green's function
    """
    z_rel = z / half_bandwidth
    return 2. / half_bandwidth * z_rel * (1 - np.sqrt(1 - 1 / (z_rel * z_rel)))
