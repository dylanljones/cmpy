# coding: utf-8
"""
Created on 25 Apr 2020
author: Dylan Jones
"""
import numpy as np
from lattpy import chain
from pyplot import Plot


def dispersion_dos_plot(omega, bins, dos):
    plot = Plot(newax=False)
    plot.set_gridspec(1, 2, wr=(3, 1))

    ax1 = plot.add_gridsubplot(0)
    plot.set_labels(r'$q$', r'$\omega(q)$')
    plot.plot(omega)
    plot.grid()

    plot.add_gridsubplot(1)
    plot.yaxis.set_ticklabels([])
    plot.grid()
    plot.plot(dos, bins)
    plot.fill(bins, 0, dos, invert_axis=True, alpha=0.5)

    plot.set_limits(x=(0, plot.xlim[1]), y=ax1.get_ylim())
    plot.set_labels(r'$n(\omega)$')
    return plot


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
