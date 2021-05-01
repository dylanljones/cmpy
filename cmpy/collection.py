# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import numpy as np
import logging

logging.captureWarnings(True)


si = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
pauli = si, sx, sy, sz


# =========================================================================
# Functions
# =========================================================================


def fermi_func(e, mu=0., beta=np.inf):
    if beta == np.inf:
        return np.heaviside(mu - e, 0)
    return 1 / (np.exp(beta * (e - mu)) + 1)


def bose_func(e, mu=0., beta=np.inf):
    if beta == np.inf:
        return np.zeros_like(e)
    return 1 / (np.exp(beta * (e - mu)) - 1)


def gaussian(x, x0=0.0, sigma=1.0):
    return np.exp(-np.power(x - x0, 2.) / (2 * np.power(sigma, 2.)))


def delta_func(x, x0, width=1, gauss=False):
    if gauss:
        sig = np.abs(x[width] - x[0])
        return np.exp(-np.power(x - x0, 2.) / (2 * np.power(sig, 2.)))
    else:
        idx = np.abs(x - x0).argmin()
        indices = np.arange(max(idx-width, 0), min(idx+width+1, len(x)-1))
        delta = np.zeros_like(x)
        delta[indices] = 1
        return delta
