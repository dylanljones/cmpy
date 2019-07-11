# -*- coding: utf-8 -*-
"""
Created on  06 2019
author: dylan

project: cmpy2
version: 1.0
"""
import numpy as np
from sciutils import Plot, Matrix


class HamiltonianCache:

    def __init__(self):
        self._ham = None

        self._memory = None
        self._mask = None

    def __bool__(self):
        return self._ham is not None

    def load(self, ham):
        self._ham = ham
        self._memory = ham.copy()
        mask = Matrix.zeros(ham.n, dtype="bool")
        mask.fill_diag(True)
        for i in range(1, ham.max_offdiag_number+1):
            mask.fill_diag(True, offset=i)
            mask.fill_diag(True, offset=-i)
        self._mask = mask

    def read(self):
        return self._ham

    def reset(self):
        self._ham[self._mask] = self._memory[self._mask]

    def clear(self):
        self._ham = None
        self._memory = None
        self._mask = None

    def __str__(self):
        string = "HamiltonianCache: "
        if not self:
            string += "empty"
        else:
            string += "\nHamiltonain:\n" + str(self._ham)
            string += "\nMemory:\n" + str(self._memory)
            string += "\nMask:\n" + str(self._mask)
        return string


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

