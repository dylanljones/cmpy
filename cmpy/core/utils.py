# -*- coding: utf-8 -*-
"""
Created on 5 Dec 2018
@author: Dylan Jones

project: cmpy
version: 1.0

Generic utilities
=================

"""
import numpy as np

eta = 1e-7j


def normalize(array):
    """ Normalizes a given array

    Parameters
    ----------
    array: array_like
        Un-normalized array

    Returns
    -------
    arr_normalized: np.ndarray
    """
    return np.asarray(array) / np.linalg.norm(array, ord=1)


def vlinspace(start, stop, n=1000):
    """ Vector linspace

    Parameters
    ----------
    start: array_like
        d-dimensional start-point
    stop: array_like
        d-dimensional stop-point
    n: int, optional
        number of points, default=1000

    Returns
    -------
    vectors: np.ndarray
    """
    axes = [np.linspace(start[i], stop[i], n) for i in range(len(start))]
    return np.asarray(axes).T


def vrange(axis_ranges):
    """ Vector range

    Parameters
    ----------
    axis_ranges: array_like
        ranges for each axis

    Returns
    -------
    vectors: np.ndarray
    """
    axis = np.meshgrid(*axis_ranges)
    grid = np.asarray([np.asarray(a).flatten("F") for a in axis]).T
    n_vecs = list(grid)
    n_vecs.sort(key=lambda x: x[0])
    return n_vecs


def in_range(x, lim):
    """ Checks if value is between limits

    Parameters
    ----------
    x: float or int
        value to check
    lim: array_like
        lower and upper boundary

    Returns
    -------
    in_range: bool
    """
    return (lim[0] <= x) and (x <= lim[1])


def in_vrange(v, lims):
    """ Checks if vector is between limits

    Parameters
    ----------
    v: array_like
        vector to check
    lims: array_like
        lower and upper boundaries

    Returns
    -------
    in_vrange: bool
    """
    return all([in_range(v[i], lims[i]) for i in range(len(v))])


def chain(items, cycle=False):
    """ Create chain between items

    Parameters
    ----------
    items: array_like
        items to join to chain
    cycle: bool, optional
        cycle to the start of the chain if True, default: False

    Returns
    -------
    chain: list
        chain of items

    Example
    -------
    >>> print(chain(["x", "y", "z"]))
    [['x', 'y'], ['y', 'z']]

    >>> print(chain(["x", "y", "z"], True))
    [['x', 'y'], ['y', 'z'], ['z', 'x']]
    """
    result = list()
    for i in range(len(items)-1):
        result.append([items[i], items[i+1]])
    if cycle:
        result.append([items[-1], items[0]])
    return result


def iter_indices(n_vecs, n_alpha):
    """ Iterate over range of indices

    Parameters
    ----------
    n_vecs: array_like
        translation vectors
    n_alpha: array_like
        atom-indices

    Returns
    -------
    idx: np.ndarray
    """
    indices = list()
    for n in n_vecs:
        for alpha in range(n_alpha):
            indices.append((n, alpha))
    return indices


def index_array(n_vecs, n_alpha):
    """ Iterate over range of indices and store as single array

    Parameters
    ----------
    n_vecs: array_like
        translation vectors
    n_alpha: array_like
        atom-indices

    Returns
    -------
    idx: np.ndarray
    """
    indices = list()
    for n in n_vecs:
        for alpha in range(n_alpha):
            indices.append([*n, alpha])
    return np.array(indices)


def translate(vectors, n):
    """ Calculate the position from the main lattice coefficients

    Parameters
    ----------
    vectors: array_like
        lattice vectors
    n: array_like:
        lattice coefficients

    Returns
    -------
    position: np.ndarray
    """
    return np.asarray(vectors) @ n
    # return np.sum(np.asarray(n) * np.asarray(vectors), axis=1)


def distance(r1, r2):
    """ calculate distance bewteen two points
    Parameters
    ----------
    r1: array_like
        first point
    r2: array_like
        second point

    Returns
    -------
    distance: float
    """
    diff = np.abs(np.asarray(r1) - np.asarray(r2))
    return np.sqrt(np.dot(diff, diff))


def neighor_n_vecs(n_vec, num=1):
    """ Find n-coefficients of neighboring cells
    Parameters
    ----------
    n_vec: array_like
        n-coefficients of cell
    num: int
        maximum neighbor distance

    Returns
    -------
    ranges: np.ndarray
        vector-range
    only_pos: bool, optional
        default is False
    """
    ranges = list()
    for n_i in n_vec:
        n0 = n_i-num
        n1 = n_i+num+1
        ranges.append(np.arange(n0, n1))
    return vrange(ranges)


def shuffle(obj, disorder):
    """ Shuffle elements of object

    Parameters
    ----------
    obj: array_like or int
        object to shuffle
    disorder: float
        disorder amount

    Returns
    -------
    shuffled: array_like
    """

    if len(obj.shape) == 0:
        return obj + np.random.uniform(-disorder/2, +disorder/2)
    else:
        shuffled = obj.copy()
        delta = np.random.uniform(-disorder/2, +disorder/2)
        for i in range(shuffled.shape[0]):
            shuffled[i, i] += delta
        return shuffled


def adj(array):
    """ Calculate the adjoint of a matrix

    Parameters
    ----------
    array: array_like

    Returns
    -------
    arr_adj: array_like
    """
    return np.conj(array).T
