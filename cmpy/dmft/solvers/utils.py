# -*- coding: utf-8 -*-
"""
Created on 15 Aug 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np


def matsubara_freq(beta, n, fermions=True):
    """ Initializes the matsubara frequency array

    Parameters
    ----------
    beta: float
        Inverse Temperature of the system (coldness)
    n: int
        Number of frequency samples
    fermions: bool, default: True
        Consider as fermionic particles

    Returns
    -------
    freq: ndarray
    """
    return np.pi * (int(fermions) + 2 * np.arange(n)) / beta


def imag_time(beta, n):
    """ Initializes the imaginary time array

    Parameters
    ----------
    beta: float
        Inverse Temperature of the system (coldness)
    n: int
        Number of time samples

    Returns
    -------
    tau: np.ndarray
    """
    return np.arange(0, beta, beta / n)


def time_freq_arrays(beta, n, factor=2., fermions=True):
    """

    Parameters
    ----------
    beta: float
        Inverse Temperature of the system (coldness)
    n: int
        Number of samples
    factor: float, default: 2.
        Factor for the number of imaginary time samples.
        The default is twice as many samples for best results.
    fermions: bool, default: True
        Consider as fermionic particles

    Returns
    -------
    tau: np.ndarray
    freq: ndarray
    """
    freq = matsubara_freq(beta, n, fermions)
    tau = imag_time(beta, int(factor * n))
    return tau, freq


class BaseSolver:

    def __init__(self, *args, **kwargs):
        pass

    def solve(self, gf_bath_inv):
        gf_imp = 1/gf_bath_inv
        return gf_imp
