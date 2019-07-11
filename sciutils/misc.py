# -*- coding: utf-8 -*-
"""
Created on  06 2019
author: dylan

project: sciutils
version: 1.0
"""
import time
import numpy as np


class OptionError(ValueError):

    def __init__(self, x=None, allowed=None, name="Value"):
        msg = f"Invalid {name}"
        if x is not None:
            msg += f"({x})"
        msg += "!"
        if allowed is not None:
            msg += f" {name} must be in {allowed}"
        super().__init__(msg)


class Cache:

    def __init__(self, data=None):
        self.data = None
        if data is not None:
            self.load(data)

    def __bool__(self):
        return self.data is not None

    def __eq__(self, other):
        return self.data == other

    def load(self, data):
        self.data = data

    def read(self):
        return self.data

    def clear(self):
        self.data = None


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


def update_mean(mean, num, x):
    """ Calculate average iteratively

    Parameters
    ----------
    mean: float
        old mean-value
    num: int
        number of iteration
    x: float
        new value

    Returns
    -------
    mena: float
    """
    return mean + (x - mean) / num


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


class Timer:

    def __init__(self):
        self._t0 = 0

    @staticmethod
    def time():
        return time.perf_counter()

    def start(self):
        self._t0 = self.time()

    @staticmethod
    def sleep(secs):
        time.sleep(secs)

    @property
    def t0(self):
        return self._t0

    @property
    def seconds(self):
        return self.time() - self._t0

    @property
    def millis(self):
        return 1000 * self.seconds


class MovingAverage:

    def __init__(self, size=100, weights=None):
        self._i = 0
        self.mem = np.zeros(size)
        self.weights = np.ones(size) if weights is None else np.asarray(weights)

    def update(self, value):
        self._i += 1
        self.mem = np.roll(self.mem, 1)
        self.mem[0] = value
        return self.get()

    def _get_memory(self):
        if self._i > self.mem.shape[0]:
            return self.mem * self.weights
        else:
            return self.mem[:self._i]

    def get(self):
        mem = self._get_memory()
        return np.mean(mem, axis=0) if len(mem) > 0 else 0


class List2D:

    def __init__(self, inputarr=None):
        if inputarr is None:
            inputarr = [[]]
        self.arr = inputarr

    @property
    def shape(self):
        return len(self), len(self[0])

    @classmethod
    def empty(cls, n, m=None, val=None):
        if m is None:
            m = n
        arr = [[val for _ in range(m)] for _ in range(n)]
        return cls(arr)

    def add_row(self, values=None):
        m = self.shape[1]
        if values is None:
            row = [None] * m
        elif not hasattr(values, "__len__"):
            row = [values] * m
        else:
            row = values
        self.arr.append(row)

    def add_collumn(self, values=None):
        n = self.shape[0]
        if values is None:
            col = [None] * n
        elif not hasattr(values, "__len__"):
            col = [values] * n
        else:
            col = values
        for i in range(n):
            self.arr[i].append(col[i])

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.arr[item]
        elif isinstance(item, tuple):
            i, j = item
            return self.arr[i][j]

    def __setitem__(self, item, value):
        if isinstance(item, int):
            if len(value) != self.shape[0]:
                raise ValueError(f"Shape mismatch ({len(value)}!={self.shape[0]})")
            self.arr[item] = value
        elif isinstance(item, tuple):
            i, j = item
            self.arr[i][j] = value

    def __call__(self, *args, **kwargs):
        return self.arr

    def __str__(self):
        string = "\n".join([str(self[i]) for i in range(self.shape[0])])
        return string
