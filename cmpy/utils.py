# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2020, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import time
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


# =========================================================================
# Misc
# =========================================================================


def chain(items, cycle=False):
    """ Create chain between items

    Parameters
    ----------
    items: array_like
        items to join to chain
    cycle: bool, optional
        cycle to the start of the chain if True, default: False

    Returns
    -------
    chain: list
        chain of items

    Example
    -------
    >>> print(chain(["x", "y", "z"]))
    [['x', 'y'], ['y', 'z']]

    >>> print(chain(["x", "y", "z"], True))
    [['x', 'y'], ['y', 'z'], ['z', 'x']]
    """
    result = list()
    for i in range(len(items)-1):
        result.append([items[i], items[i+1]])
    if cycle:
        result.append([items[-1], items[0]])
    return result


def running_mean(a, window=3, pad_edges=True):
    weights = np.repeat(1.0, window) / window
    if pad_edges:
        a = np.pad(a, (window, window), mode='edge')
        return np.convolve(a, weights, 'same')[window:-window]
    else:
        return np.convolve(a, weights, 'same')


def frmt_time(seconds: float, short: bool = False, width: int = 0) -> str:
    """ Return a formated string for a given time in seconds.

    Parameters
    ----------
    seconds: float
        Time value to format
    short: bool, optional
        Flag if short representation should be used.
    width: int, optional
        Optional minimum length of the returned string

    Returns
    -------
    time_str: str
    """

    string = "00:00"

    # short time string
    if short:
        if seconds > 0:
            mins, secs = divmod(seconds, 60)
            if mins > 60:
                hours, mins = divmod(mins, 60)
                string = f"{hours:02.0f}:{mins:02.0f}h"
            else:
                string = f"{mins:02.0f}:{secs:02.0f}"

    # Full time strings
    else:
        if seconds < 1e-3:
            nanos = 1e6 * seconds
            string = f"{nanos:.0f} \u03BCs"
        elif seconds < 1:
            millis = 1000 * seconds
            string = f"{millis:.1f} ms"
        elif seconds < 60:
            string = f"{seconds:.1f} s"
        else:
            mins, seconds = divmod(seconds, 60)
            if mins < 60:
                string = f"{mins:.0f}:{seconds:04.1f} min"
            else:
                hours, mins = divmod(mins, 60)
                string = f"{hours:.0f}:{mins:02.0f}:{seconds:02.0f} h"

    if width > 0:
        string = f"{string:>{width}}"
    return string


class Timer:

    __slots__ = ["_time", "_t0"]

    def __init__(self, method=time.perf_counter):
        self._time = method
        self._t0 = 0

    @property
    def seconds(self):
        return self.time() - self._t0

    @property
    def millis(self):
        return 1000 * (self.time() - self._t0)

    def time(self):
        return self._time()

    def start(self):
        self._t0 = self.time()

    def eta(self, progress: float) -> float:
        if not progress:
            return 0.0
        else:
            return (1 / progress - 1) * self.time()

    def strfrmt(self, short: bool = False, width: int = 0) -> str:
        return frmt_time(self.time(), short, width)

    @staticmethod
    def sleep(t):
        time.sleep(t)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.strfrmt(short=True)})'

    def __str__(self) -> str:
        return self.strfrmt(short=True)
