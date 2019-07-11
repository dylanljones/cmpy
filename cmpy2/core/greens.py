# -*- coding: utf-8 -*-
"""
Created on 14 Feb 2019
author: Dylan

project: cmpy2
version: 1.0
"""
import numpy as np
from scipy import linalg as la


def greens(ham, omega):
    """ Calculate the Green's function for a given hamiltonian

    Parameters
    ----------
    omega: float
        energy value to calculate the Green's function
    ham: array_like
        hamiltonian
    Returns
    -------
    gf: array_like
    """
    e = np.eye(ham.shape[0]) * omega
    return la.inv(e - ham, overwrite_a=True)


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
    eye = np.eye(ham.shape[0])
    alpha = t
    beta = np.conj(t).T
    h_l = ham.copy()
    h_b = ham.copy()
    h_r = ham.copy()
    while True:
        # calculate greens-functions
        gf_l = la.inv(omega * eye - h_l)
        gf_b = la.inv(omega * eye - h_b)
        gf_r = la.inv(omega * eye - h_r)

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


def rgf(ham, omega):
    """ Recursive green's function

    Calculate green's function using the recursive green's function formalism.

    Parameters
    ----------
    ham: Hamiltonian
        hamiltonian of model, must allready be blocked
    omega: float
        energy-value to calculate the Green's functions

    Returns
    -------
    gf_1n: array_like
        lower left block of greens function
    """

    e = np.eye(ham.block_size[0]) * omega
    n_blocks = ham.block_shape[0]

    g_nn = la.inv(e - ham.get_block(0, 0), overwrite_a=True)
    g_1n = g_nn
    for i in range(1, n_blocks):
        h = ham.get_block(i, i) + ham.get_block(i, i-1) @ g_nn @ ham.get_block(i-1, i)
        g_nn = la.inv(e - h, overwrite_a=True)
        g_1n = g_1n @ ham.get_block(i-1, i) @ g_nn
    return g_1n
