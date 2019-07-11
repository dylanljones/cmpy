# -*- coding: utf-8 -*-
"""
Created on  06 2019
author: dylan

project: cmpy2
version: 1.0
"""
from sciutils import fit, build_fit


def transfunc(x, l, y0):
    return - (2 / l) * x + y0


def localization_length(x, y, l0=100):
    popt, errs = fit(transfunc, x, y, [l0, 0])
    return popt[0], errs[0]


def localization_length_built(x, y, l0=100):
    popt, errs = fit(transfunc, x, y, [l0, 0])
    loclen = popt[0], errs[0]
    data = build_fit(transfunc, x, popt)
    return loclen, data
