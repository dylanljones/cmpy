# -*- coding: utf-8 -*-
"""
Created on 14 Aug 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from sciutils import Plot


def bath_greens_function_inv(gf_loc, sigma=0):
    """ Inverse of the bath green's function of the system

    Parameters
    ----------
    gf_loc: complex ndarray or complex
        Local green's function of the lattice
    sigma: float ndarray or float

    Returns
    -------
    gf_bath_inv: ndarray or float
        Inverse value of the bath green's function
    """
    return 1/gf_loc + sigma


def bath_greens_function(gf_loc, sigma=0):
    """ Bath green's function of the system

    Parameters
    ----------
    gf_loc: complex ndarray or complex
        Local green's function of the lattice
    sigma: float ndarray or float

    Returns
    -------
    gf_bath: ndarray or float
        Value of the bath green's function
    """
    return 1/bath_greens_function_inv(gf_loc, sigma)


def quasiparticle_weight(sigma, beta):
    """ Calculates the impurity quasiparticle weight from the self energy in the matsubara frequencies

    Parameters
    ----------
    sigma: np.ndarray
        Self energy of the impurity in matsubara frequencies
    beta: float
        Inverse Temperature (coldness)

    Returns
    -------
    z: float
    """
    sigma = sigma.imag
    if sigma[1] > sigma[0]:
        return 0.
    dw = np.pi / beta
    dsigma0 = sigma[0] / dw
    return 1 / (1 - dsigma0)


def self_energy(gf_bath_inv, gf_imp_inv):
    return gf_bath_inv - gf_imp_inv
