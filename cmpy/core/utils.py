# coding: utf-8
"""
Created on 07 Apr 2020
author: Dylan Jones
"""
import time
import numpy as np
from enum import Enum


class Ascii(Enum):

    alpha   = 'α'
    beta    = 'β'
    gamma   = 'γ'
    delta   = 'δ'
    epsilon = 'ϵ'
    zeta    = 'ζ'
    eta     = 'η'
    theta   = 'θ'
    iota    = 'ι'
    kappa   = 'κ'
    lamb    = 'λ'
    mu      = 'μ'

    omega   = 'ω'

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


# =========================================================================
# Misc
# =========================================================================


def running_mean(a, window=3, pad_edges=True):
    weights = np.repeat(1.0, window) / window
    if pad_edges:
        a = np.pad(a, (window, window), mode='edge')
        return np.convolve(a, weights, 'same')[window:-window]
    else:
        return np.convolve(a, weights, 'same')


class Timer:

    def __init__(self, method=time.perf_counter):
        self._time = method
        self._t0 = 0

    @property
    def seconds(self):
        return self.time() - self._t0

    @property
    def millis(self):
        return 1000 * self.seconds

    def time(self):
        return self._time()

    def start(self):
        self._t0 = self.time()

    @staticmethod
    def sleep(t):
        time.sleep(t)


class FunctionContainer:

    def __init__(self, x, y=None):
        x = np.asarray(x)
        if y is None:
            y = np.zeros_like(x)
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    @property
    def n(self):
        return self.x.shape[0]

    @property
    def shape(self):
        return self.y.shape

    @property
    def real(self):
        return self.y.real

    @property
    def imag(self):
        return self.y.imag

    @property
    def xlim(self):
        return np.min(self.x), np.max(self.x)

    @property
    def ylim(self):
        return np.min(self.y, axis=0), np.max(self.y, axis=0)

    def __repr__(self):
        return f"{self.__class__.__name__}(shape: {self.shape})"

    def __getitem__(self, item):
        return self.y[item]

    def __setitem__(self, key, value):
        self.y[key] = value
