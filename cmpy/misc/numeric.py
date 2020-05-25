# coding: utf-8
"""
Created on 07 Apr 2020
author: Dylan Jones
"""
import numpy as np


def running_mean(a, window=3, pad_edges=True):
    weights = np.repeat(1.0, window) / window
    if pad_edges:
        a = np.pad(a, (window, window), mode='edge')
        return np.convolve(a, weights, 'same')[window:-window]
    else:
        return np.convolve(a, weights, 'same')


def gaussian(x, x0=0.0, sigma=1.0):
    return np.exp(-np.power(x - x0, 2.) / (2 * np.power(sigma, 2.)))


def delta_function(x, x0, width=1, gauss=False):
    if gauss:
        sig = np.abs(x[width] - x[0])
        return np.exp(-np.power(x - x0, 2.) / (2 * np.power(sig, 2.)))
    else:
        idx = np.abs(x - x0).argmin()
        indices = np.arange(max(idx-width, 0), min(idx+width+1, len(x)-1))
        delta = np.zeros_like(x)
        delta[indices] = 1
        return delta
