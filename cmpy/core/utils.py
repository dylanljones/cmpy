# -*- coding: utf-8 -*-
"""
Created on  06 2019
author: dylan

project: cmpy
version: 1.0
"""
import numpy as np
import scipy.linalg as la
from scitools import Plot

paulix = np.array([[0, 1], [1, 0]])
pauliy = np.array([[0, -1j], [1j, 0]])
pauliz = np.array([[1, 0], [0, -1]])

# =========================================================================
#                               GENERAL
# =========================================================================


def get_eta(omegas, n):
    dw = abs(omegas[1] - omegas[0])
    return 1j * n * dw


def get_omegas(omax=5, n=1000, deta=0.5):
    omegas = np.linspace(-omax, omax, n)
    return omegas, get_eta(omegas, deta)


def ensure_array(x):
    return x if hasattr(x, "__len__") else [x]


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


# =========================================================================
#                               FERMIONS
# =========================================================================


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


def diagonalize(operator):
    """diagonalizes single site Spin Hamiltonian"""
    eig_values, eig_vecs = la.eigh(operator)
    eig_values -= np.amin(eig_values)
    return eig_values, eig_vecs


def partition_func(beta, energies):
    return np.exp(-beta*energies).sum()


def expectation(eigvals, eigstates, operator, beta):
    ew = np.exp(-beta * eigvals)
    aux = np.einsum('i,ji,ji', ew, eigstates, operator.dot(eigstates))
    return aux / ew.sum()


# =========================================================================
#                               PLOTS
# =========================================================================


def plot_greens_function(omegas, gf):
    gf = np.asarray(gf)
    ymax = np.max(-gf.imag)
    ylim = (0, 1.1 * ymax)

    if len(gf) == 2:
        gf_up, gf_dn = gf

        plot = Plot(create=False)
        plot.set_gridspec(2, 1)

        ax = plot.add_gridsubplot(0)
        # plot.set_limits(ylim=ylim)
        plot.plotfill(omegas, -gf_up.imag)
        plot.set_labels(ylabel=r"A$_{\uparrow}$")
        plot.grid()

        plot.add_gridsubplot(1, sharex=ax)

        plot.invert_yaxis()
        plot.plotfill(omegas, -gf_dn.imag)
        plot.set_labels(xlabel=r"$\omega$", ylabel=r"A$_{\downarrow}$")
        plot.grid()

        plot.set_limits(xlim=0)
        plot.label_outer()
    else:
        plot = Plot(xlabel=r"$\omega$", ylabel=r"A")
        plot.plotfill(omegas, -gf.imag)
        plot.set_limits(xlim=0)
        plot.grid()

    plot.tight()
    return plot
