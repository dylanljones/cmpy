# -*- coding: utf-8 -*-
"""
Created on  06 2019
author: dylan

project: cmpy
version: 1.0
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_bands(band_sections, point_names, show=True):
    # Build bands
    bands = band_sections[0]
    ticks = [0, bands.shape[0]]
    for i in range(1, len(band_sections)):
        bands = np.append(bands, band_sections[i], axis=0)
        ticks.append(bands.shape[0])
    ticklabels = list(point_names) + [point_names[0]]

    # Plot
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E(k)$")
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    for x in ticks[1:-1]:
        ax.axvline(x, lw=0.5, color="0.5")
    ax.axhline(0, lw=0.5, color="0.5")
    ax.plot(bands)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax
