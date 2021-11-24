# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Methods and objects for handling dense and sparse matrices."""

import itertools
import numpy as np
from numpy.lib.stride_tricks import as_strided  # noqa
from scipy import linalg as la
import scipy.sparse
from typing import NamedTuple
from functools import partial
from matplotlib import colors
import matplotlib.pyplot as plt
import colorcet as cc

__all__ = ["transpose", "matshow", "hermitian", "is_hermitian", "diagonal", "fill_diagonal",
           "decompose", "reconstruct", "Decomposition", "EigenState", "Matrix"]

transpose = partial(np.swapaxes, axis1=-2, axis2=-1)


# =============================================================================
# Plotting
# =============================================================================

class MidpointNormalize(colors.Normalize):

    """Mid-point colormap normalization

    References
    ----------
    https://stackoverflow.com/a/50003503
    """

    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # calculates the values of the colors vmin and vmax are assigned to
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) /
                                                 (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) /
                                                 (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        result, is_scalar = self.process_value(value)
        # data values
        x = [self.vmin, self.midpoint, self.vmax]
        # color values assigned to data values
        y = [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y), mask=result.mask)


def matshow(mat, show=True, cmap=cc.m_coolwarm, colorbar=False, values=False,
            xticklabels=None, yticklabels=None, ticklabels=None, xrotation=45,
            normoffset=0.2, normcenter=0, ax=None):
    """Plots a two dimensional array.

    Parameters
    ----------
    mat : array_like
        The matrix to plot.
    show : bool, optional
        if True, call plt.show(), default: True
    colorbar : bool, optional
        Show colorbar if True.
    values : bool, optional
        if True, print values in boxes
    cmap : str, optional
        colormap used in the plot
    xticklabels : list, optional
        Optional labels of the right basis states of the matrix, default: None
    yticklabels : list, optional
        Optional labels of the left basis states of the matrix, default: None
    ticklabels : list, optional
        Optional ticklabels for setting both axis ticks instead of using
        x_ticklabels and x_ticklabels seperately. The default is None.
    xrotation : int, optional
        Amount of rotation of the x-labels, default: 45
    normoffset : float, optional
        Offset of norm used for colormap.
    normcenter : float or None, optional
        The center of the colormap norm. If `None`, the colormap will not be centered!
    ax : plt.Axes, optional
        Axes item
    """
    mat = np.asarray(mat)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.xaxis.set_label_position("top")

    cmap = cmap or cc.m_coolwarm
    nlim = np.min(mat), np.max(mat)
    off = normoffset * abs(nlim[1] - nlim[0])
    vmin, vmax = nlim[0] - off, nlim[1] + off
    if normcenter is None:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=normcenter)

    im = ax.matshow(mat, cmap=cmap, norm=norm)

    if values:
        dec = 1
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if val:
                    center = np.array([i, j])
                    ax.text(*center, s=f"{val:.{dec}f}", va="center", ha="center")

    if colorbar:
        fig.colorbar(im, ax=ax)

    if max(mat.shape) < 20:
        ax.set_xticks(np.arange(0, mat.shape[0], 1))
        ax.set_yticks(np.arange(0, mat.shape[1], 1))
        if ticklabels is not None:
            xticklabels = yticklabels = ticklabels
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels, rotation=xrotation, ha="right")
        if yticklabels is not None:
            ax.set_yticklabels(yticklabels)

    fig.tight_layout()

    if show:
        plt.show()

    return ax


# =============================================================================
# Methods
# =============================================================================


def hermitian(a):
    """Returns the hermitian of an array.

    Parameters
    ----------
    a : array_like
        The input array.

    Returns
    -------
    hermitian : np.ndarray
    """
    return np.conj(np.transpose(a))


def is_hermitian(a, rtol=1e-5, atol=1e-8, equal_nan=False):
    """Checks if an array is hermitian.

    Checks if the hermitian of the given array equals the array within
    a tolerance by calling np.allcloe.

    Parameters
    ----------
    a : array_like
        The array to check.
    rtol : float, optional
        The relative tolerance parameter used in the comparison (see np.allclose).
    atol : float, optional
        The absolute tolerance parameter used in the comparison (see np.allclose).
    equal_nan : bool, optional
        Flag if ``NaN``'s are compared.

    Returns
    -------
    hermitian : bool
    """
    return np.allclose(a, np.conj(np.transpose(a)), rtol, atol, equal_nan)


def diagonal(mat, offset=0):
    """Gets the optionally offset diagonal elements of an array.

    Parameters
    ----------
    mat : np.ndarray
        input matrix
    offset : int, optional
        offset index of diagonal. An offset of ``1`` results in the first upper
        offdiagonal of the matrix, an offset of -1 in the first lower offdiagonal.
        The default is ``0``.

    Returns
    -------
    diag : np.ndarray
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
    """Fills the optionally offset diagonal of an array.

    Parameters
    ----------
    mat : np.ndarray
        Array whose diagonal is to be filled, it gets modified in-place.
    val : scalar or array_like
        Value to be written on the diagonal, its type must be compatible
        with that of the array a.
    offset : int, optional
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


def decompose(arr, h=None):
    """Computes the eigen-decomposition of a matrix.

    Finds the eigenvalues and the left and right eigenvectors of a matrix. If the matrix
    is hermitian ``eigh`` is used for computing the right eigenvectors.

    Parameters
    ----------
    arr : (N, N) array_like
        A complex or real matrix whose eigenvalues and eigenvectors will be computed.
    h : bool, optional
        Flag if matrix is hermitian. If ``None`` the matrix is checked. This determines the
        scipy method used for computing the eigenvalue problem.

    Returns
    -------
    vr : (N, N) np.ndarray
        The right eigenvectors of the matrix.
    xi : (N, ) np.ndarray
        The eigenvalues of the matrix.
    vl : (N, N) np.ndarray
        The left eigenvectors of the matrix.
    """
    if h is None:
        h = is_hermitian(arr)
    if h:
        xi, vr = np.linalg.eigh(arr)
        vl = np.conj(vr.T)
    else:
        xi, vl, vr = scipy.linalg.eig(arr, left=True, right=True)
    return vr, xi, vl


def reconstruct(rv, xi, rv_inv, method='full'):
    """Computes a matrix from an eigen-decomposition.

    The matrix is reconstructed using eigenvalues and left and right eigenvectors.

    Parameters
    ----------
    rv : (N, N) np.ndarray
        The right eigenvectors of a matrix.
    xi : (N, ) np.ndarray
        The eigenvalues of a matrix
    rv_inv : (N, N) np.ndarray
        The left eigenvectors of a matrix.
    method : str, optional
        The mode for reconstructing the matrix. If mode is 'full' the original matrix
        is reconstructed, if mode is 'diag' only the diagonal elements are computed.
        The default is 'full'.

    Returns
    -------
    arr : (N, N) np.ndarray
        The reconstructed matrix.
    """
    method = method.lower()
    if 'diag'.startswith(method):
        arr = ((transpose(rv_inv) * rv) @ xi[..., np.newaxis])[..., 0]
    elif 'full'.startswith(method):
        arr = (rv * xi[..., np.newaxis, :]) @ rv_inv
    else:
        arr = np.einsum(method, rv, xi, rv_inv)
    return arr


class Decomposition:
    """Eigen-decomposition of a matrix.

    Parameters
    ----------
    rv : (N, N) np.ndarray
        The right eigenvectors of a matrix.
    xi : (N, ) np.ndarray
        The eigenvalues of a matrix
    rv_inv : (N, N) np.ndarray
        The left eigenvectors of a matrix.
    """
    __slots__ = ('rv', 'xi', 'rv_inv')

    def __init__(self, rv, xi, rv_inv):
        self.rv = rv
        self.xi = xi
        self.rv_inv = rv_inv

    @classmethod
    def decompose(cls, arr, h=None):
        """Computes the decomposition of a matrix and initializes a ``Decomposition``-instance.

        Parameters
        ----------
        arr : (N, N) array_like
            A complex or real matrix whose eigenvalues and eigenvectors will be computed.
        h : bool, optional
            Flag if matrix is hermitian. If ``None`` the matrix is checked. This determines the
            scipy method used for computing the eigenvalue problem.

        Returns
        -------
        decomposition : Decomposition
        """
        rv, xi, rv_inv = decompose(arr, h)
        return cls(rv, xi, rv_inv)

    def reconstrunct(self, xi=None, method='full'):
        """Computes a matrix from the eigen-decomposition.

        Parameters
        ----------
        xi : (N, ) array_like, optional
            Optional eigenvalues to compute the matrix. If ``None`` the eigenvalues of the
            decomposition are used. The default is ``None``.
        method : str, optional
        The mode for reconstructing the matrix. If mode is 'full' the original matrix
        is reconstructed, if mode is 'diag' only the diagonal elements are computed.
        The default is 'full'.

        Returns
        -------
        arr : (N, N) np.ndarray
            The reconstructed matrix.
        """
        xi = self.xi if xi is None else xi
        return reconstruct(self.rv, xi, self.rv_inv, method)

    def normalize(self):
        """Normalizes the eigenstates and creates a new ``Decomposition``-instance."""
        rv = self.rv / np.linalg.norm(self.rv)
        rv_inv = self.rv_inv / np.linalg.norm(self.rv_inv)
        return self.__class__(rv, self.xi, rv_inv)

    def transform_basis(self, transformation):
        """Transforms the eigen-basis into another basis.

        Parameters
        ----------
        transformation : (N, N) array_like
            The transformation-matrix to transform the basis.
        """
        rv = self.rv @ np.asarray(transformation)
        self.rv = rv
        self.rv_inv = np.linalg.inv(rv)

    def __str__(self):
        return f"{self.__class__.__name__}[{self.rv.shape}x{self.xi.shape}x{self.rv_inv.shape}]"


class EigenState(NamedTuple):

    energy: float = np.infty
    state: np.ndarray = None
    n_up: int = None
    n_dn: int = None


# =============================================================================
# Matrix object
# =============================================================================


class Matrix(np.ndarray):
    """Matrix-object based on ``np.ndarray``."""

    def __new__(cls, inputarr, dtype=None) -> 'Matrix':
        """ Initialize Matrix

        Parameters
        ----------
        inputarr : array_like
            Input array for the Matrix
        dtype : str or np.dtype, optional
            Optional datatype of the matrix
        """
        if isinstance(inputarr, scipy.sparse.spmatrix):
            inputarr = inputarr.toarray()
        self = np.asarray(inputarr, dtype).view(cls)
        if len(self.shape) != 2:
            raise ValueError(f"Inputarray must be 2 dimensional, not {len(self.shape)}D!")
        return self

    @classmethod
    def zeros(cls, *shape, dtype=None) -> 'Matrix':
        """Initializes the ``Matrix`` filled with zeros.

        Parameters
        ----------
        shape : tuple or int
            Positional shape arguments of the matrix. This can either be one or two
            integers or a shape tuple. If only one value is given the resulting
            matrix will be square.
        dtype : str or np.dtype, optional
            Optional datatype of the matrix
        """
        if len(shape) == 1:
            shape = shape[0] if hasattr(shape[0], "__len__") else (shape[0], shape[0])
        return cls(np.zeros(shape), dtype)

    @classmethod
    def zeros_like(cls, a, dtype=None) -> 'Matrix':
        """Initializes the ``Matrix`` filled with zeros based on the given array.

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
        """Initializes the ``Matrix`` filled with a specific value

        Parameters
        ----------
        shape : tuple or int
            Positional shape arguments of the matrix. This can either be one or two
            integers or a shape tuple. If only one value is given the resulting
            matrix will be square.
        value : float or complex, optional
            The fill value of the matrix.
        dtype : str or np.dtype, optional
            Optional datatype of the matrix.
        """
        if len(shape) == 1:
            shape = shape[0] if hasattr(shape[0], "__len__") else (shape[0], shape[0])
        return cls(np.full(shape, value), dtype)

    @classmethod
    def eye(cls, n, dtype=None) -> 'Matrix':
        """Initializes the `` Matrix`` as identity matrix.

        Parameters
        ----------
        n : int
            Size of the square matrix
        dtype : str or np.dtype, optional
            Optional datatype of the matrix
        """
        return cls(np.eye(n), dtype)

    @classmethod
    def uniform(cls, *shape, low=0., high=1., dtype=None) -> 'Matrix':
        """Initialize the ``Matrix`` filled with random values.

        Parameters
        ----------
        shape : tuple or int
            Positional shape arguments of the matrix. This can either be one or two
            integers or a shape tuple. If only one value is given the resulting
            matrix will be square.
        low : float, optional
            Lower boundary of the output interval
        high : float, optional
            Upper boundary of the output interval
        dtype : str or np.dtype, optional
            Optional datatype of the matrix
        """
        if len(shape) == 1:
            shape = shape[0] if hasattr(shape[0], "__len__") else (shape[0], shape[0])
        return cls(np.random.uniform(low, high, shape), dtype)

    def show(self, show=True, **kwargs) -> plt.Axes:
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

    def almost_equal(self, other, rtol: float = 1e-5, atol: float = 1e-8,
                     equal_nan: bool = False) -> bool:
        """Checks if the matrix is almost equal to an other array."""
        return np.allclose(self, other, rtol, atol, equal_nan)

    def inv(self) -> 'Matrix':
        """ Matrix: Inverse of the Matrix """
        return self.__class__(la.inv(self))

    def diag(self, offset=0):
        """Gets the diagonal elements (with an optional offset) of the matrix

        Parameters
        ----------
        offset : int, optional
            offset index of diagonal. An offset of 1 results
            in the first upper offdiagonal of the matrix,
            an offset of -1 in the first lower offdiagonal.

        Returns
        -------
        np.ndarray
        """
        return diagonal(self, offset)

    def fill_diag(self, val, offset=0):
        """Fill the diagonal elements (with an optional offset) of the given array.

        Parameters
        ----------
        val : scalar or array_like
            Value to be written on the diagonal, its type must be compatible
            with that of the array a.
        offset: int or tuple, optional
            Offset index of diagonal. An offset of 1 results
            in the first upper offdiagonal of the matrix,
            an offset of -1 in the first lower offdiagonal.
        """
        offset = np.atleast_1d(offset)
        for off in offset:
            fill_diagonal(self, val, off)

    def eig(self, check_hermitian=True):
        """Calculate eigenvalues and -vectors of the Matrix-instance.

        Parameters
        ----------
        check_hermitian : bool, optional
            If True and the instance of the the matrix is hermitian,
            ``np.eigh`` is used as eigensolver.

        Returns
        -------
        eigenvalues : np.ndarray
            eigenvalues of the matrix
        eigenvectors : np.ndarray
            eigenvectors of the matrix
        """
        if check_hermitian and self.is_hermitian:
            return la.eigh(self)
        else:
            return la.eig(self)

    def eigh(self):
        """Calculate eigenvalues and -vectors of the hermitian matrix.

        Returns
        -------
        eigenvalues : np.ndarray
            eigenvalues of the matrix
        eigenvectors : np.ndarray
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
        self[i:i + size_x, j:j + size_y] = array

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
        self[i:i + size_x, j:j + size_y] += array.astype(self.dtype)

    def block(self, r, c, block=(2, 2)):
        """Returns a block of the matrix.

        Parameters
        ----------
        r : int
            Row of the block.
        c : int
            Column of the block.
        block : tuple, optional
            The shape of the blocks.

        Returns
        -------
        block : np.ndarray
        """
        row = slice(r, r + block[0])
        col = slice(c, c + block[0])
        return self[row, col]

    def blocks(self, block=(2, 2)):
        """Returns the matrix as block-matrix.

        Parameters
        ----------
        block : tuple, optional
            The shape of the blocks.

        Returns
        -------
        block_array: np.ndarray
        """
        shape = (int(self.shape[0] / block[0]), int(self.shape[1] / block[1])) + block
        strides = (block[0] * self.strides[0], block[1] * self.strides[1]) + self.strides
        return as_strided(self, shape=shape, strides=strides)  # noqa
