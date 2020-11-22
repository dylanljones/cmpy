# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2020, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import itertools
import numpy as np
from numpy.lib.stride_tricks import as_strided  # noqa
from scipy import linalg as la
from functools import partial
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import colorcet as cc


transpose = partial(np.swapaxes, axis1=-2, axis2=-1)


class MatrixPlot:

    DEFAULT_CMAP = cc.m_coolwarm

    def __init__(self, cmap=None, norm_offset=0.0):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        cmap = cmap or self.DEFAULT_CMAP
        self.cmap = cm.get_cmap(cmap)
        self.ax.xaxis.set_label_position('top')

        self.array = None
        self.im = None
        self.norm = None
        self.norm_offset = norm_offset

    def matshow(self, array, symmetric_norm=False):
        array = np.asarray(array)
        array = array.real + array.imag
        nlim = np.min(array), np.max(array)
        off = self.norm_offset * abs(nlim[1] - nlim[0])
        if symmetric_norm:
            nrange = max(nlim) + off
            norm = colors.Normalize(vmin=-nrange, vmax=nrange)
        else:
            norm = colors.Normalize(vmin=nlim[0] - off, vmax=nlim[1] + off)

        self.array = array
        self.norm = norm
        self.im = self.ax.matshow(array, cmap=self.cmap, norm=self.norm)

    def set_labels(self, row=None, col=None):
        if col is not None:
            self.ax.set_xlabel(col)
        if row is not None:
            self.ax.set_ylabel(row)

    def set_tickstep(self, step=1):
        self.ax.set_xticks(np.arange(0, self.array.shape[0], step))
        self.ax.set_yticks(np.arange(0, self.array.shape[1], step))

    def set_ticklabels(self, xlabels=None, ylabels=None, xrotation=45):
        if xlabels is not None:
            self.ax.set_xticks(np.arange(0, self.array.shape[0], 1))
            self.ax.set_xticklabels(xlabels, rotation=xrotation, ha="right")

        if ylabels is not None:
            self.ax.set_yticks(np.arange(0, self.array.shape[1], 1))
            self.ax.set_yticklabels(ylabels)

    def show_values(self):
        dec = 1
        for i in range(self.array.shape[0]):
            for j in range(self.array.shape[1]):
                val = self.array[i, j]
                if val:
                    center = np.array([i, j])
                    self.ax.text(*center, s=f"{val:.{dec}f}", va="center", ha="center")

    def draw_grid(self, color="black"):
        self.ax.set_axisbelow(below_axis=False)
        self.ax.grid(which="minor", color=color)

    def grid(self):
        self.ax.grid()

    def show_colorbar(self):
        self.fig.colorbar(self.im, ax=self.ax)

    def tight(self, *args, **kwargs):
        self.fig.tight_layout(*args, **kwargs)

    def show(self, tight=True):
        if tight:
            self.fig.tight_layout()
        plt.show()


def matshow(mat, show=True, cmap=cc.m_coolwarm, norm_offset=0.2, colorbar=False, values=False,
            x_ticklabels=None, y_ticklabels=None, ticklabels=None, xrotation=45):
    """ Plot the matrix using the MatrixPlot object

    Parameters
    ----------
    mat: np.ndarray
        The matrix to plot.
    show: bool, optional
        if True, call plt.show(), default: True
    colorbar: bool, optional
        Show colorbar if True.
    values: bool, optional
        if True, print values in boxes
    cmap: str, optional
        colormap used in the plot
    x_ticklabels: list, optional
        Optional labels of the right basis states of the matrix, default: None
    y_ticklabels: list, optional
        Optional labels of the left basis states of the matrix, default: None
    ticklabels: list, optional
        Optional ticklabels for setting both axis ticks instead of using
        x_ticklabels and x_ticklabels seperately. The default is None.
    xrotation: int, optional
        Amount of rotation of the x-labels, default: 45
    norm_offset: float, optional
        Offset of norm used for colormap.
    """
    mp = MatrixPlot(cmap=cmap, norm_offset=norm_offset)
    mp.matshow(mat)
    if values:
        mp.show_values()
    if colorbar:
        mp.show_colorbar()
    if max(mat.shape) < 20:
        mp.set_tickstep(1)
        if ticklabels is not None:
            x_ticklabels = y_ticklabels = ticklabels
        mp.set_ticklabels(x_ticklabels, y_ticklabels, xrotation)
    if show:
        mp.show()
    return mp


# =========================================================================
# METHODS
# =========================================================================

def hermitian(a):
    """ Returns the hermitian of the array

    Parameters
    ----------
    a: array_like

    Returns
    -------
    hermitian: np.ndarray
    """
    return np.conj(np.transpose(a))


def is_hermitian(a, rtol=1e-5, atol=1e-8, equal_nan=False):
    """ Returns True if the array is hermitian and False otherwise.

    Checks if the hermitian of the given array equals the array within a tolerance by calling np.allcloe.

    Parameters
    ----------
    a: array_like
        The array to check.
    rtol: float, optional
        The relative tolerance parameter used in the comparison (see np.allclose).
    atol: float, optional
        The absolute tolerance parameter used in the comparison (see np.allclose).
    equal_nan: bool, optional
        Flag if NaN's are compared.

    Returns
    -------
    hermitian: bool
    """
    return np.allclose(a, np.conj(np.transpose(a)), rtol, atol, equal_nan)


def diagonal(mat, offset=0):
    """ Get the optionally offset diagonal elements of a matrix

    Parameters
    ----------
    mat: np.ndarray
        input matrix
    offset: int, default: 0
        offset index of diagonal. An offset of 1 results
        in the first upper offdiagonal of the matrix,
        an offset of -1 in the first lower offdiagonal.

    Returns
    -------
    np.ndarray
    """
    if offset:
        n, m = mat.shape
        if offset > 0:
            # Upper off-diagonal
            idx = slice(0, n), slice(abs(offset), m)
        else:
            # Lower off-diagonal
            idx = slice(abs(offset), n), slice(0, m)
        diag = np.diag(mat[idx])
    else:
        diag = np.diag(mat)
    return np.asarray(diag)


def fill_diagonal(mat, val, offset=0):
    """ Fill the optionally offset diagonal of the given array

    Parameters
    ----------
    mat: np.ndarray
        Array whose diagonal is to be filled, it gets modified in-place.
    val: scalar or array_like
        Value to be written on the diagonal, its type must be compatible
        with that of the array a.
    offset: int, default: 0
        offset index of diagonal. An offset of 1 results
        in the first upper offdiagonal of the matrix,
        an offset of -1 in the first lower offdiagonal.
    """
    if offset:
        n, m = mat.shape
        if offset > 0:
            # Upper off-diagonal
            idx = slice(0, n), slice(abs(offset), m)
        else:
            # Lower off-diagonal
            idx = slice(abs(offset), n), slice(0, m)
        np.fill_diagonal(mat[idx], val)

    else:
        np.fill_diagonal(mat, val)


class Decomposition:

    __slots__ = ('rv', 'xi', 'rv_inv')

    def __init__(self, rv, xi, rv_inv):
        self.rv = rv
        self.xi = xi
        self.rv_inv = rv_inv

    def transform_basis(self, transformation):
        rv = self.rv @ np.asarray(transformation)
        self.rv = rv
        self.rv_inv = np.linalg.inv(rv)

    @classmethod
    def decompose(cls, a):
        a = np.asarray(a)
        if is_hermitian(a):
            xi, rv = np.linalg.eigh(a)
            rv_inv = np.conj(rv.T)
        else:
            xi, rv = np.linalg.eig(a)
            rv_inv = np.linalg.inv(rv)
        return cls(rv, xi, rv_inv)

    def reconstrunct(self, xi=None, method='full'):
        method = method.lower()
        xi = self.xi if xi is None else xi
        if 'diag'.startswith(method):
            a = ((transpose(self.rv_inv)*self.rv) @ xi[..., np.newaxis])[..., 0]
        elif 'full'.startswith(method):
            a = (self.rv * xi[..., np.newaxis, :]) @ self.rv_inv
        else:
            a = np.einsum(method, self.rv, xi, self.rv_inv)
        return a

    def __str__(self):
        return f"{self.__class__.__name__}[{self.rv.shape}x{self.xi.shape}x{self.rv_inv.shape}]"


# =========================================================================
# MATRIX OBJECT
# =========================================================================


class Matrix(np.ndarray):

    def __new__(cls, inputarr, dtype=None) -> 'Matrix':
        """ Initialize Matrix

        Parameters
        ----------
        inputarr: array_like
            Input array for the Matrix
        dtype: str or np.dtype, optional
            Optional datatype of the matrix
        """
        self = np.asarray(inputarr, dtype).view(cls)
        if len(self.shape) != 2:
            raise ValueError(f"Inputarray must be 2 dimensional, not {len(self.shape)}D!")
        return self

    @classmethod
    def zeros(cls, *shape, dtype=None) -> 'Matrix':
        """Initializes the ``Matrix`` filled with zeros.

        Parameters
        ----------
        shape: tuple or int
            Positional shape arguments of the matrix. This can either be one or two integers or a shape tuple.
            If only one value is given the resulting matrix will be square.
        dtype: str or np.dtype, optional
            Optional datatype of the matrix
        """
        if len(shape) == 1:
            shape = shape[0] if hasattr(shape[0], "__len__") else (shape[0], shape[0])
        return cls(np.zeros(shape), dtype)

    @classmethod
    def zeros_like(cls, a, dtype=None) -> 'Matrix':
        """ Initialize Matrix filled with zeros based on the given array

        Parameters
        ----------
        a: array_like
            The shape and dtype of this array will be copied.
        dtype: str or np.dtype, optional
            Optional datatype of the matrix
        """
        return cls(np.zeros_like(a), dtype)

    @classmethod
    def full(cls, *shape, value=1.0, dtype=None) -> 'Matrix':
        """ Initialize Matrix filled with a specific value

        Parameters
        ----------
        shape: tuple or int
            Positional shape arguments of the matrix. This can either be one or two integers or a shape tuple.
            If only one value is given the resulting matrix will be square.
        value: float or complex, optional
            The fill value of the matrix.
        dtype: str or np.dtype, optional
            Optional datatype of the matrix
        """
        if len(shape) == 1:
            shape = shape[0] if hasattr(shape[0], "__len__") else (shape[0], shape[0])
        return cls(np.full(shape, value), dtype)

    @classmethod
    def eye(cls, n, dtype=None) -> 'Matrix':
        """ Initialize Matrix as unitary matrix

        Parameters
        ----------
        n: int
            size of the square matrix
        dtype: str or np.dtype, optional
            Optional datatype of the matrix
        """
        return cls(np.eye(n), dtype)

    @classmethod
    def uniform(cls, *shape, low=0., high=1., dtype=None) -> 'Matrix':
        """ Initialize Matrix filled with random values

        Parameters
        ----------
        shape: tuple or int
            Positional shape arguments of the matrix. This can either be one or two integers or a shape tuple.
            If only one value is given the resulting matrix will be square.
        low: float, default: 0
            Lower boundary of the output interval
        high: float, default: 1
            Upper boundary of the output interval
        dtype: str or np.dtype, optional
            Optional datatype of the matrix
        """
        if len(shape) == 1:
            shape = shape[0] if hasattr(shape[0], "__len__") else (shape[0], shape[0])
        return cls(np.random.uniform(low, high, shape), dtype)

    def show(self, show=True, **kwargs) -> MatrixPlot:
        """ Plot the matrix using the MatrixPlot object.

        See Also
        --------
        matshow
        """
        return matshow(self, show, **kwargs)

    def __str__(self):
        prod = itertools.product(range(self.shape[0]), range(self.shape[0]))
        x = max([len(str(self[i, j])) for i, j in prod])
        string = ""
        for i in range(self.shape[0]):
            line = "["
            for j in range(self.shape[1]):
                val = self[i, j]
                if np.imag(val) == 0:
                    s = str(np.real(val)) + " "
                elif np.real(val) == 0:
                    s = str(np.imag(val)) + "j"
                else:
                    s = str(val)
                line += f"{s:^{x}} "
            string += line[:-1] + "]\n"
        return string[:-1]

    # =========================================================================

    @property
    def h(self) -> 'Matrix':
        """Returns the conplex conjugate of the mtrix."""
        return np.conj(self).T

    @property
    def is_hermitian(self) -> bool:
        """Checks if the matrix is hermitian"""
        return self.almost_equal(np.conj(self).T)

    def equal(self, other) -> bool:
        """Checks if the matrix is equal to an other array."""
        return np.array_equal(self, other)

    def almost_equal(self, other, rtol: float = 1e-5, atol: float = 1e-8, equal_nan: bool = False) -> bool:
        """Checks if the matrix is almost equal to an other array."""
        return np.allclose(self, other, rtol, atol, equal_nan)

    def inv(self) -> 'Matrix':
        """ Matrix: Inverse of the Matrix """
        return self.__class__(la.inv(self))

    def diag(self, offset=0):
        """Gets the diagonal elements (with an optional offset) of the matrix

        Parameters
        ----------
        offset: int, default: 0
            offset index of diagonal. An offset of 1 results
            in the first upper offdiagonal of the matrix,
            an offset of -1 in the first lower offdiagonal.

        Returns
        -------
        np.ndarray
        """
        return diagonal(self, offset)

    def fill_diag(self, val, offset=0):
        """ Fill the diagonal elements (with an optional offset) of the given array

        Parameters
        ----------
        val: scalar or array_like
            Value to be written on the diagonal, its type must be compatible
            with that of the array a.
        offset: int or tuple, default: 0
            Offset index of diagonal. An offset of 1 results
            in the first upper offdiagonal of the matrix,
            an offset of -1 in the first lower offdiagonal.
        """
        offset = np.atleast_1d(offset)
        for off in offset:
            fill_diagonal(self, val, off)

    def eig(self, check_hermitian=True):
        """ Calculate eigenvalues and -vectors of the Matrix-instance.

        Parameters
        ----------
        check_hermitian: bool, optional
            If True and the instance of the the mnatrix is hermitian, np.eigh is used as eigensolver.

        Returns
        -------
        eigenvalues: np.ndarray
            eigenvalues of the matrix
        eigenvectors: np.ndarray
            eigenvectors of the matrix
        """
        if check_hermitian and self.is_hermitian:
            return la.eigh(self)
        else:
            return la.eig(self)

    def eigh(self):
        """ Calculate eigenvalues and -vectors of the hermitian matrix.

        Returns
        -------
        eigenvalues: np.ndarray
            eigenvalues of the matrix
        eigenvectors: np.ndarray
            eigenvectors of the matrix
        """
        return la.eigh(self)

    def eigvals(self, check_hermitian=True, num_range=None):
        """ np.ndarray: The eigenvalues of the matrix """
        if check_hermitian and self.is_hermitian:
            return la.eigvalsh(self, eigvals=num_range)
        else:
            return la.eigvals(self)

    def eigvecs(self, check_hermitian=True):
        """ np.ndarray: The eigenvectors of the matrix """
        return self.eig(check_hermitian)[1]

    def decompose(self):
        """ Decomposes the matrix into it's eigen-decomposition (eigenvalues and eigenvectors).

        Returns
        -------
        decomposition: Decomposition
        """
        return Decomposition.decompose(self)

    @classmethod
    def reconstruct(cls, decomposition, xi=None, method='full'):
        return cls(decomposition.reconstrunct(xi, method))

    def insert(self, i, j, array):
        """ Insert subarray starting at index (i, j)

        Parameters
        ----------
        i: int
            row index to start inserting the subarray
        j: int
            collumn index to start inserting the subarray
        array: array_like
            subarray to insert into matrix
        """
        try:
            size_x, size_y = array.shape
        except ValueError:
            size_x, size_y = array.shape[0], 1
        self[i:i+size_x, j:j+size_y] = array

    def add(self, i, j, array):
        """ Add subarray starting at index (i, j)

        Parameters
        ----------
        i: int
            row index to start adding the subarray
        j: int
            collumn index to start adding the subarray
        array: array_like
            subarray to add to matrix
        """
        try:
            size_x, size_y = array.shape
        except ValueError:
            size_x, size_y = array.shape[0], 1
        self[i:i+size_x, j:j+size_y] += array.astype(self.dtype)

    def block(self, r, c, block=(2, 2)):
        row = slice(r, r + block[0])
        col = slice(c, c + block[0])
        return self[row, col]

    def blocks(self, block=(2, 2)):
        shape = (int(self.shape[0] / block[0]), int(self.shape[1] / block[1])) + block
        strides = (block[0] * self.strides[0], block[1] * self.strides[1]) + self.strides
        return as_strided(self, shape=shape, strides=strides)
