# -*- coding: utf-8 -*-
"""
Created on  06 2019
author: dylan

project: cmpy
version: 1.0
"""
import numpy as np
from sciutils import Plot


def fermi_dist(energy, beta, mu=1):
    """ Calculates the fermi-distributions for fermions

    Parameters
    ----------
    energy: float nd.ndarray or float
        The energy value
    beta: float
        Coldnes (inverse temperature)
    mu: float, default=0
        Chemical potential. At T=0 this is the Fermi-energy E_F

    Returns
    -------
    fermi: float np.ndarray
    """
    exponent = np.asarray((energy - mu) * beta).clip(-1000, 1000)
    return 1. / (np.exp(exponent) + 1)


def spectral(gf):
    return -1/np.pi * gf.imag


def uniform(w, size):
    return np.random.uniform(-w/2, w/2, size=size)


def uniform_eye(w, size):
    return np.eye(size) * np.random.uniform(-w / 2, w / 2, size=size)


def plot_bands(band_sections, point_names, show=True):
    # Build bands
    bands = band_sections[0]
    ticks = [0, bands.shape[0]]
    for i in range(1, len(band_sections)):
        bands = np.append(bands, band_sections[i], axis=0)
        ticks.append(bands.shape[0])
    ticklabels = list(point_names) + [point_names[0]]

    # Plot
    plot = Plot()
    plot.plot(bands)
    plot.set_labels(r"$k$", r"$E(k)$")
    plot.set_limits(xlim=(ticks[0], ticks[-1]))
    plot.set_ticks(xticks=ticks)
    plot.set_ticklabels(xticks=ticklabels)
    plot.draw_lines(x=ticks[1:-1], y=0, lw=0.5, color="0.5")
    plot.tight()
    if show:
        plot.show()
    return plot
