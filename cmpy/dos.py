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
from typing import Union

__all__ = ["get_bin_borders", "get_bins", "density_of_states"]


def get_bin_borders(amin: float, amax: float, bins: Union[int, float],
                    padding: Union[float, tuple] = 0.0) -> np.ndarray:
    """Creates an array of bin-borders, seperating the region of each bin.

    Parameters
    ----------
    amin : float
        The lower boundary of the bins.
    amax : float
        The upper boundary of the bins.
    bins : int or float
        Argument for spacing the bins. ``bins`` is interpreted as the numbe rof bins
        if an ``int`` is passed and as the bin-spacing if a ``float`` is passed.
    padding : float or array_like, optional
        Optional padding of the bins.

    Returns
    -------
    borders: np.ndarray
        An array of N+1 borders for N bins.
    """
    if isinstance(padding, (int, float)):
        padding = [padding, padding]

    if padding:
        amin -= padding[0]
        amax += padding[1]

    if isinstance(bins, int):
        datarange = abs(amax - amin)
        size = datarange / bins
        borders = np.arange(amin, amax, size)
    elif isinstance(bins, float):
        borders = np.arange(amin, amax, bins)
    else:
        borders = np.array(bins)
        diffs = np.unique(np.diff(borders))
        assert np.allclose(diffs[0], diffs[1:])

    # Ensure first and last bin are in data-limits
    borders[0] = amin
    if borders[-1] != amax:
        borders = np.append(borders, amax)
    # Even out the size of the first and last bin
    size0 = borders[1] - borders[0]
    size1 = borders[-1] - borders[-2]
    delta = (size0 - size1) / 2
    borders[1:-1] -= delta

    # diffs = np.unique(np.diff(borders))
    # assert np.allclose(diffs[0], diffs[1:])

    return borders


def get_bins(borders: np.ndarray, loc: float = 0.5) -> np.ndarray:
    """Returns the positions of the bins defined by bin-borders.

    Parameters
    ----------
    borders : (N+1) np.ndarray
        An array containing the borders of the bins.
    loc : float, optional
        Position of the bins. If ``0.`` the position is the left border of the bins,
        if ``1.`` the right border. The default is ``0.5`` (center of the bin-borders).

    Returns
    -------
    bin_positions: (N) np.ndarray
        The positions of the bins.
    """
    delta = borders[1] - borders[0]
    return delta * loc + borders[0] + np.append(0, np.cumsum(np.diff(borders))[:-1])


def density_of_states(disp, bins=None, loc=0.5, normalize=False, counts=False,
                      border=False, padding=0.):
    """Computes the density of states.

    Parameters
    ----------
    disp : array_like
        Array of the single or multi-band dispersion.
    bins : int or float, optional
        Bin argument. The argument is interpreted as number of bins (int) or binwidth (float).
        The default is the number of samples divided by 100.
    loc : float, optional
        Relative location of the value of the bin. The default is the center.
    normalize : bool, optional
        Flag if the density of states is normalized to `1`. The default is ``False``.
    counts : bool, optional
        Flag if the counts of the density of states are returned. The default is ``False``.
    border : bool, optional
        If ``True`` a zero is added to both sides of the dos.
    padding : float or array_like, optional
        Padding used in binning.

    Returns
    -------
    bins: (N) np.ndarray
        The bin positions of the density of states.
    dos: (N) np.ndarray
        The density of states normed according to the given arguments (normalize, counts).
    """
    # Prepare dispersion data
    disp = np.atleast_2d(disp)
    if disp.shape[0] > disp.shape[1]:
        disp = np.swapaxes(disp, 0, 1)
    num_bands, num_disp = disp.shape

    # Initialize bins
    if bins is None:
        bins = int(num_disp / 100)
    amin, amax = np.min(disp), np.max(disp)
    borders = get_bin_borders(amin, amax, bins, padding)
    binvals = get_bins(borders, loc)
    num_bins = len(binvals)

    # Compute density of states
    state_counts = np.zeros((num_bands, num_bins))
    for band, disparray in enumerate(disp):
        disparray = np.sort(disparray)
        for i in range(num_bins):
            x0, x1 = borders[i], borders[i + 1]
            states = np.where((x0 <= disparray) & (disparray <= x1))[0]
            state_counts[band, i] = len(states)

    # Assert counts match number of states
    state_counts = np.sum(state_counts, axis=0)
    assert int(np.sum(state_counts)) == disp.size

    # Normalize state-count
    if normalize:
        norm = np.sum(state_counts)
    elif counts:
        norm = 1.
    else:  # Actual norm of dos
        domega = binvals[1] - binvals[0]
        norm = num_disp * domega
    state_counts = state_counts / norm

    # Add zeros to the side of the dos
    if border:
        step = 0.01 * abs(binvals[1] - binvals[0])
        binvals = np.append(binvals[0] - step, np.append(binvals, binvals[-1] + step))
        state_counts = np.append(0, np.append(state_counts, 0))

    return binvals, state_counts
