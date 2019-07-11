# -*- coding: utf-8 -*-
"""
Created on  06 2019
author: dylan

project: sciutils
version: 1.0
"""
import numpy as np
from scipy.optimize import curve_fit


def fit(func, x, y, p0=None):
    popt, pcov = curve_fit(func, x, y, p0=p0)
    errs = np.sqrt(np.diag(pcov))
    return popt, errs


def build_fit(func, x, popt):
    return func(x, *popt)


def fit_and_build(func, x, y, p0=None):
    popt, errs = fit(func, x, y, p0)
    xfit, yfit = build_fit(func, x, popt)
    return (popt, errs), (xfit, yfit)


# =========================================================================
# Standard functions
# =========================================================================


def linear_func(x, m=1., y0=0.):
    return m*x + y0


def quadratic_func(x, x0=0., xs=1., y0=0., ys=1.):
    return ys * (xs * (x - x0))**2 + y0


def exp_func(x, x0=0., xs=1., y0=0., ys=1.):
    return ys * np.exp(xs * (x - x0)) + y0
