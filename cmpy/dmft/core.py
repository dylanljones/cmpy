# -*- coding: utf-8 -*-
"""
Created on 14 Aug 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np


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


def dmft_step(omega, sigma, solver):
    gf_loc = bethe_gf_omega(omega - sigma)
    gf_bath_inv = bath_greens_function_inv(gf_loc, sigma)

    gf_imp = solver.solve(gf_bath_inv)

    sigma_imp = self_energy(gf_bath_inv, 1 / gf_imp)
    return sigma_imp
