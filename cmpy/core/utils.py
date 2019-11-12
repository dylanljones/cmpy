# -*- coding: utf-8 -*-
"""
Created on  06 2019
author: dylan

project: cmpy
version: 1.0
"""
import numpy as np
from scitools import Plot


def get_eta(omegas, n):
    dw = abs(omegas[1] - omegas[0])
    return 1j * n * dw


def get_omegas(omax=5, n=1000, deta=0.5):
    omegas = np.linspace(-omax, omax, n)
    return omegas, get_eta(omegas, deta)


def ensure_array(x):
    return x if hasattr(x, "__len__") else [x]


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
