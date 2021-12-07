# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import numpy as np
from typing import Union

si = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
sp = 0.5 * (sx + 1j * sy)
sm = 0.5 * (sx - 1j * sy)
pauli = si, sx, sy, sz


def kron(*args) -> np.ndarray:
    """Computes the Kronecker product of two or more arrays.

    Parameters
    ----------
    *args : list of array_like or array_like
        Input arrays.

    Returns
    -------
    out : np.ndarray
        The Kronecker product of the input arrays.

    Examples
    --------
    >>> kron([1, 0], [1, 0])
    array([1, 0, 0, 0])
    >>> kron([[1, 0], [1, 0]])
    array([1, 0, 0, 0])
    >>> kron([1, 0], [1, 0], [1, 0])
    array([1, 0, 0, 0, 0, 0, 0, 0])
    """
    if len(args) == 1:
        args = args[0]
    x = 1
    for arg in args:
        x = np.kron(x, arg)
    return x


def density_matrix(psi: np.ndarray) -> np.ndarray:
    r"""Computes the density matrix ρ of a given state-vector |ψ>.
    .. math::
        ρ = |ψ><ψ|
    Parameters
    ----------
    psi: (N) np.ndarray
        The input state-vector ψ.
    Returns
    -------
    rho : (N, N) np.ndarray
        The density matrix ρ.
    Examples
    --------
    >>> density_matrix(np.array([1, 0]))
    array([[1, 0],
           [0, 0]])
    >>> density_matrix(np.array([1, 0]))
    array([[1, 0],
           [0, 0]])
    >>> density_matrix(np.array([1j, 0]))
    array([[1.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j]])
    """
    psiv = np.atleast_2d(psi).T
    # psiv = psi[:, np.newaxis]
    return np.dot(psiv, np.conj(psiv.T))


# =========================================================================
# Functions
# =========================================================================


def fermi_func(e, mu=0.0, beta=np.inf):
    if beta == np.inf:
        return np.heaviside(mu - e, 0)
    return 1 / (np.exp(beta * (e - mu)) + 1)


def bose_func(e, mu=0.0, beta=np.inf):
    if beta == np.inf:
        return np.zeros_like(e)
    return 1 / (np.exp(beta * (e - mu)) - 1)


def gaussian(x: np.ndarray, x0: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Gaussian probability distribution

    Parameters
    ----------
    x : (N, ) np.ndarray
        The x-values for evaluating the function.
    x0 : float, optional
        The center of the Gaussian function. The default is the origin.
    sigma : float, optional
        The width of the probability distribution.

    Returns
    -------
    psi : (N, ) np.ndarray
    """
    # return np.exp(-np.power(x - x0, 2.) / (2 * np.power(sigma, 2.)))
    psi = np.exp(-np.power(x - x0, 2.0) / (4 * sigma ** 2))
    norm = np.sqrt(np.sqrt(2 * np.pi) * sigma)
    return psi / norm


def delta_func(x: np.ndarray, x0: float = 0) -> np.ndarray:
    """Delta distribution.

    Parameters
    ----------
    x : (N, ) np.ndarray
        The x-values for evaluating the function.
    x0 : float, optional
        The center of the function. The default is the origin.
    """
    idx = np.abs(x - x0).argmin()
    delta = np.zeros_like(x)
    delta[idx] = 1.0
    return delta


def initialize_state(size, index=None, mode="delta", *args, **kwargs):
    """Initialize the statevector of a state

    Parameters
    ----------
    size : int
        The size of the statevector/Hilbert space.
    index : int or float, optional
        The position or site index where the state is localized.
    mode : str, optional
        The mode for initializing the state. valid modes are: 'delta' and 'gauss'
    *args
        Positional arguments for state function.
    **kwargs
        Keyword arguments for the state function.

    Returns
    -------
    psi : (N, ) np.ndarray
    """
    x = np.arange(size)
    index = int(size // 2) if index is None else index
    if mode == "delta":
        psi = delta_func(x, index)
    elif mode == "gauss":
        psi = gaussian(x, index, *args, **kwargs)
    else:
        raise ValueError(f"Mode '{mode}' not supported. Valid modes: 'delta', 'gauss'")
    return psi


def tevo_state_eig(eigvals, eigvecs, state, times):
    """Evolve a state under the given hamiltonian.

    Parameters
    ----------
    eigvals : (N) np.ndarray
    eigvecs : (N, N) np.ndarray
    state : (N, ) np.ndarray
    times : float or (M, ) array_like

    Returns
    -------
    states : (M, N) np.nd_array or (N, ) np.ndarray
    """
    scalar = not hasattr(times, "__len__")
    if scalar:
        times = [times]
    eigvecs_t = eigvecs.T

    # Project initial state into eigenbasis
    proj = np.inner(eigvecs_t, state)
    # Evolve projected states
    proj_t = proj[np.newaxis, ...] * np.exp(-1j * eigvals * times[:, np.newaxis])
    # Reconstruct the new state in the site-basis
    states = np.dot(proj_t, eigvecs_t)

    return states[0] if scalar else states


def tevo_state(ham, state, times):
    """Evolve a state under the given hamiltonian.

    Parameters
    ----------
    ham : (N, N) np.ndarray
    state : (N, ) np.ndarray
    times : float or (M, ) array_like

    Returns
    -------
    states : (M, N) np.nd_array or (N, ) np.ndarray
    """
    eigvals, eigvecs = np.linalg.eigh(ham)
    return tevo_state_eig(eigvals, eigvecs, state, times)


# =========================================================================
# Density of states
# =========================================================================


def get_bin_borders(
    amin: float,
    amax: float,
    bins: Union[int, float],
    padding: Union[float, tuple] = 0.0,
) -> np.ndarray:
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


def density_of_states(
    disp, bins=None, loc=0.5, normalize=False, counts=False, border=False, padding=0.0
):
    """Computes the density of states.

    Parameters
    ----------
    disp : array_like
        Array of the single or multi-band dispersion.
    bins : int or float, optional
        Bin argument. The argument is interpreted as number of bins (int)
        or binwidth (float). The default is the number of samples divided by 100.
    loc : float, optional
        Relative location of the value of the bin. The default is the center.
    normalize : bool, optional
        Flag if the density of states is normalized to `1`. The default is `False`.
    counts : bool, optional
        Flag if the counts of the density of states are returned.
        The default is `False`.
    border : bool, optional
        If ``True`` a zero is added to both sides of the dos.
    padding : float or array_like, optional
        Padding used in binning.

    Returns
    -------
    bins: (N) np.ndarray
        The bin positions of the density of states.
    dos: (N) np.ndarray
        The density of states normed according to the arguments (normalize, counts).
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
        norm = 1.0
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


def histogram_median(hist, dp=31.7 / 2):
    median = np.percentile(hist, 50, axis=0)
    hist_up = np.percentile(hist, 100 - dp, axis=0)
    hist_dn = np.percentile(hist, dp, axis=0)
    return median, np.abs(median - [hist_dn, hist_up])
