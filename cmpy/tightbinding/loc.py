# -*- coding: utf-8 -*-
"""
Created on 1 Apr 2018
@author: Dylan Jones
project: cmpy
version: 1.0
"""
import os
from os.path import abspath, dirname, join
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from cmpy.core import eta, Progress

DEFAULT_MODE = "lin"


def _exp_func(x, l, a):
    return a * np.exp(-x/l)


def _lin_func(x, l, y0):
    return - 1/l * x + y0


def fit(lengths, trans, p0=None, n_fit=20, mode=DEFAULT_MODE):
    if p0 is None:
        p0 = [50, 1]
    lengths = lengths[-n_fit:]
    trans = trans[-n_fit:]
    if mode == "lin":
        func = _lin_func
    else:
        func = _exp_func
    popt, pcov = curve_fit(func, lengths, trans, p0=p0)
    return popt, np.sqrt(np.diag(pcov))


def build_fit(lengths, popt, n_fit=20, mode=DEFAULT_MODE):
    lengths = lengths[-n_fit:]
    y = None
    if mode == "exp":
        y = _exp_func(lengths, *popt)
    elif mode == "lin":
        y = _lin_func(lengths, *popt)
    return lengths, y


def loc_length(lengths, trans, p0=None, n_fit=20, mode=DEFAULT_MODE):
    # Fit data, starting from "fit_start"
    popt, errs = fit(lengths, trans, p0, n_fit, mode)
    # return localization-length and -error
    return popt[0], errs[0]


def loc_length_fit(lengths, trans, p0=None, n_fit=20, mode=DEFAULT_MODE):
    # Fit data, starting from "fit_start"
    popt, errs = fit(lengths, trans, p0, n_fit, mode)
    # return localization-length and -error
    return popt[0], errs[0], build_fit(lengths, popt, n_fit, mode)

# =============================================================================
