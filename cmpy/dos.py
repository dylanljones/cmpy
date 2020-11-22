# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2020, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import numpy as np
import matplotlib.pyplot as plt
from .utils import chain


def scale_xaxis(num_points, disp, scales=None):
    sect_size = len(disp) / (num_points - 1)
    scales = np.ones(num_points - 1) if scales is None else scales
    k0, k, ticks = 0, list(), [0]
    for scale in scales:
        k.extend(k0 + np.arange(sect_size) * scale)
        k0 = k[-1]
        ticks.append(k0)
    return k, ticks


def band_subplots(ticks, labels, x_label="k", disp_label="E(k)", grid="both"):
    fig, ax = plt.subplots()
    ax.set_xlim(0, ticks[-1])
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    if x_label:
        ax.set_xlabel(f"${x_label}$")
    if disp_label:
        ax.set_ylabel(f"${disp_label}$")

    if grid:
        ax.grid(axis=grid)
    fig.tight_layout()
    return fig, ax


def plot_dispersion(ax, k, disp, fill=True, alpha=0.2, lw=1.0):
    for band in disp.T:
        ax.plot(k, band, lw=lw)
        if fill:
            ax.fill_between([0, np.max(k)], min(band), max(band), alpha=alpha)


def plot_bands(disp, labels, x_label="k", disp_label="E(k)", grid="both", fill=True,
               alpha=0.2, lw=1.0, show=True, scales=None):

    num_points = len(labels)
    k, ticks = scale_xaxis(num_points, disp, scales)

    fig, ax = band_subplots(ticks, labels, x_label, disp_label, grid)
    plot_dispersion(ax, k, disp, fill, alpha, lw)

    if show:
        plt.show()
    return fig, ax


def band_dos_subplots(ticks, labels, x_label="k", disp_label="E(k)", dos_label="n(E)",
                      wratio=(3, 1), grid="both"):
    fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": wratio}, sharey="all")
    ax1, ax2 = axs

    ax1.set_xlim(0, ticks[-1])
    if x_label:
        ax1.set_xlabel(f"${x_label}$")
    if disp_label:
        ax1.set_ylabel(f"${disp_label}$")
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(labels)

    if dos_label:
        ax2.set_xlabel(f"${dos_label}$")
    ax2.set_xticks([0])

    if grid:
        ax1.grid(axis=grid)
        ax2.grid(axis=grid)
    fig.tight_layout()
    return fig, axs


def plot_band_dos(disp, bins, dos, labels, x_label="k", disp_label="E(k)", dos_label="n(E)",
                  wratio=(3, 1), grid="both", fill=True, disp_alpha=0.2, dos_color="C0",
                  dos_alpha=0.2, lw=1.0, scales=None, show=True):
    num_points = len(labels)
    k, ticks = scale_xaxis(num_points, disp, scales)

    fig, axs = band_dos_subplots(ticks, labels, x_label, disp_label, dos_label, wratio, grid)
    ax1, ax2 = axs
    plot_dispersion(ax1, k, disp, fill=fill, alpha=disp_alpha, lw=lw)
    ax2.plot(dos, bins, lw=lw, color=dos_color)
    ax2.fill_betweenx(bins, 0, dos, alpha=dos_alpha, color=dos_color)
    ax2.set_xlim(0, ax2.get_xlim()[1])

    if show:
        plt.show()
    return fig, axs


class Bins:

    def __init__(self, arg, bins, loc=0.5, padding=0.0, center=True):
        self.n = 0
        self.limits = None
        self.values = None

        self.set_bins(arg, bins, loc, padding, center)

    def set_bins(self, arg, bins, loc=0.5, padding=0.0, center=True):
        if isinstance(arg, np.ndarray):
            amin, amax = np.min(arg), np.max(arg)
        else:
            amin, amax = arg

        if padding:
            amin -= padding
            amax += padding

        if isinstance(bins, int):
            datarange = abs(amax - amin)
            size = datarange / bins
            positions = np.arange(amin, amax, size)
        elif isinstance(bins, float):
            positions = np.arange(amin, amax, bins)
        else:
            positions = bins

        # Ensure first and last bin re data-limits
        positions[0] = amin
        if positions[-1] != amax:
            positions = np.append(positions, amax)

        if center:
            # Even out the size of the first and last bin
            size0 = positions[1] - positions[0]
            size1 = positions[-1] - positions[-2]
            delta = (size0 - size1) / 2
            positions[1:-1] -= delta

        self.limits = np.array(chain(positions))
        self.n = len(self.limits)
        self.set_bin_values(loc)

    def set_bin_values(self, loc=0.0):
        self.values = np.array([x0 + (loc * (x1 - x0)/2) for x0, x1 in self.limits])


def density_of_states(dispersion, bins=None, loc=0.5, padding=0.01, normalize=True):
    """ Computes the density of states by numeric integration of the dispersion.

    Parameters
    ----------
    dispersion: array_like
        Array of the single or multi-band dispersion.
    bins: int or float, optional
        Bin argument. The argument is interpreted as number of bins (int) or binwidth (float).
        The default is the number of samples divided by 100
    loc: float, optional
        Relative location of the value of the bin. The default is the center.
    padding: float, optional
        Padding used in binning.
    normalize: bool, optional
        Flag if the density of states is nromalized. The default is True.

    Returns
    -------
    bins: (N) np.ndarray
    dos: (N) np.ndarray
    """
    if bins is None:
        bins = int(len(dispersion) / 100)
    disp = np.atleast_2d(dispersion)
    if disp.shape[0] > disp.shape[1]:
        disp = np.swapaxes(disp, 0, 1)

    bins = Bins(disp, bins, loc, padding)

    bands = len(disp)
    counts = np.zeros((bands, bins.n))
    for band, disparray in enumerate(disp):
        disparray = np.sort(disparray)
        for i in range(bins.n):
            x0, x1 = bins.limits[i]
            states = np.where((x0 <= disparray) & (disparray < x1))[0]
            counts[band, i] = len(states)

    x = bins.values
    if normalize:
        counts = counts / np.linalg.norm(counts)

    return x, np.sum(counts, axis=0)
