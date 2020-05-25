# coding: utf-8
"""
Created on 31 Mar 2020
Author: Dylan Jones
"""
import itertools
import numpy as np
from scipy import linalg as la
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import colorcet as cc


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


def offdiag_numbers(mat):
    """ Calculate the diagonal offsets of the matrix

    Parameters
    ----------
    mat

    Returns
    -------
    lower: int
        number of non-zero lower diagonals
    upper: int
        number of non-zero upper diagonals
    """
    n = mat.shape[0]
    lower, upper = n, n

    # Calculate number of non-zero lower diagonals
    for i in range(-n + 1, 0):
        if np.any(diagonal(mat, offset=i)):
            lower = -i
            break
    # Calculate number of non-zero lower diagonals
    for i in reversed(range(1, n)):
        if np.any(diagonal(mat, offset=i)):
            upper = i
            break
    return lower, upper


def max_offdiag_number(mat):
    """ Calculate the maxiamal diagonal offset of the matrix

    Parameters
    ----------
    mat

    Returns
    -------
    max_offdiag: int
        maximal number of non-zero diagonals
    """
    return max(offdiag_numbers(mat))


def ordered_diag_matrix(mat, lu=None):
    """ Convert a array to a sparse matrix in the ordered diagonal form

    Parameters
    ----------
    mat: array_like
        Input array
    lu: tuple of int, default: None
        Number of non-zero lower and upper diagonals. Value will be
        calculated if not specified.

    Returns
    -------
    ab: np.ndarray
        band matrix
    lu: tuple of int
        Number of non-zero lower and upper diagonals.
    """
    mat = np.asarray(mat)
    n = mat.shape[0]

    if lu is not None:
        lower, upper = lu
    else:
        # Calculate number of non-zero lower and upper diagonals
        lower, upper = offdiag_numbers(mat)

    # Build the band matrix
    ab = np.zeros((lower + upper + 1, n), dtype=mat.dtype)

    # Upper non-zero diagonals
    for i in range(1, upper+1):
        ab[upper - i, -(n-i):] = diagonal(mat, offset=i)
    # Main diagonal
    ab[upper] = np.diag(mat)
    # Lower non-zero diagonals
    for i in range(1, lower+1):
        ab[upper + i, :n - i] = diagonal(mat, offset=-i)

    return ab, (lower, upper)


def upper_diag_matrix(mat, upper=None):
    """ Convert a array to a sparse matrix in the upper diagonal form

    Parameters
    ----------
    mat: array_like
        Input array
    upper: int, default: None
        Number of non-zero upper diagonals. Value will be
        calculated if not specified.

    Returns
    -------
    ab: np.ndarray
        band matrix
    upper: int
        Number of non-zero lupper diagonals.
    """
    mat = np.asarray(mat)
    n = mat.shape[0]

    if upper is None:
        # Calculate number of non-zero upper diagonals
        for i in reversed(range(1, n)):
            if np.any(diagonal(mat, offset=i)):
                upper = i
                break

    # Build the band matrix
    ab = np.zeros((upper + 1, n), dtype=mat.dtype)

    # Upper non-zero diagonals
    for i in range(1, upper + 1):
        ab[upper-i, -(n - i):] = diagonal(mat, offset=i)
    # Main diagonal
    ab[-1] = np.diag(mat)

    return ab, upper


def lower_diag_matrix(mat, lower=None):
    """ Convert a array to a sparse matrix in the lower diagonal form

    Parameters
    ----------
    mat: array_like
        Input array
    lower: int, default: None
        Number of non-zero lower diagonals. Value will be
        calculated if not specified.

    Returns
    -------
    ab: np.ndarray
        band matrix
    lower: tuple of int
        Number of non-zero lower diagonals.
    """
    mat = np.asarray(mat)
    n = mat.shape[0]

    if lower is None:
        # Calculate number of non-zero lower and upper diagonals
        lower, upper = None, None
        for i in range(-n + 1, 0):
            if np.any(diagonal(mat, offset=i)):
                lower = -i
                break

    # Build the band matrix
    ab = np.zeros((lower + 1, n), dtype=mat.dtype)

    # Main diagonal
    ab[0] = np.diag(mat)
    # Lower non-zero diagonals
    for i in range(1, lower+1):
        ab[i, :n - i] = diagonal(mat, offset=-i)

    return ab, lower


def solve_banded(a, b, lu=None):
    """ Solve the equation a x = b for x, assuming a is banded matrix.

    The input matrix is converted to a diagonal ordered form which then is used to
    solve the problem using scipy.

    Parameters
    ----------
    a: array_like
        Input matrix
    b: array_like
        Right hand side
    lu: tuple of int, default: None
        Number of non-zero lower and upper diagonals. Value will be
        calculated if not specified.

    Returns
    -------
    x: np.ndarray
    """
    ab, lu = ordered_diag_matrix(a, lu)
    return la.solve_banded(lu, ab, b)


def inv_banded(a, lu=None):
    """ Compute the (multiplicative) inverse of a matrix.

    Given a square matrix a, return the matrix ainv satisfying dot(a, ainv) = dot(ainv, a) = eye(a.shape[0]).

    Parameters
    ----------
    a: array_like
        Input matrix
    lu: tuple of int, default: None
        Number of non-zero lower and upper diagonals. Value will be
        calculated if not specified.

    Returns
    -------
    ainv: np.ndarray
    """
    a = np.asarray(a)
    ab, lu = ordered_diag_matrix(a, lu)
    return la.solve_banded(lu, ab, np.eye(a.shape[0]))


def eig_banded(a, lower=False):
    """ Solve real symmetric or complex hermitian band matrix eigenvalue problem.

    Parameters
    ----------
    a: array_like
        Input array
    lower: bool, default: False
        Is the matrix in the lower form. (Default is upper form)

    Returns
    -------
    eigvals: np.ndarray
    eigvecs: np.ndarray
    """
    if lower:
        a_band = lower_diag_matrix(a)[0]
    else:
        a_band = upper_diag_matrix(a)[0]
    return la.eig_banded(a_band, lower=lower)


# =========================================================================
# MATRIX-BLOCKING
# =========================================================================


def _diag_sizes(shape, size):
    n_max = max(shape)
    if hasattr(size, "__len__"):
        total_size = sum(size)
        if total_size < n_max:
            n = int(n_max/total_size)
            diag_sizes = list(itertools.chain(*itertools.repeat(size, n)))
        else:
            diag_sizes = size
    else:
        n = int(n_max / size)
        diag_sizes = [size for _ in range(n)]
    return diag_sizes


def blockmatrix_slices(shape, size):
    n, m = shape
    block_sizes = _diag_sizes(shape, size)
    # Build row-indices
    r_indices = [0]
    for i in range(1, n+1):
        idx = sum(block_sizes[:i])
        r_indices.append(idx)

        if idx >= n:
            break
    # Build collumn-indices
    c_indices = [0]
    for i in range(1, m+1):
        idx = sum(block_sizes[:i])
        c_indices.append(idx)
        if idx >= m:
            break

    # Construct slices
    slices = list()
    for i in range(len(r_indices)-1):
        row = list()
        for j in range(len(c_indices)-1):
            idx0 = slice(r_indices[i], r_indices[i+1])
            idx1 = slice(c_indices[j], c_indices[j+1])
            row.append((idx0, idx1))
        slices.append(row)

    return slices


class MatrixBlocks(list):

    def __init__(self, slices):
        super().__init__(slices)

    @property
    def shape(self):
        return len(self), len(self[0])

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return super().__getitem__(idx)
        else:
            return super().__getitem__(idx[0])[idx[1]]


# =========================================================================
# MATRIX OBJECT
# =========================================================================


class Matrix(np.ndarray):

    def __new__(cls, inputarr, dtype=None):
        """ Initialize Matrix

        Parameters
        ----------
        inputarr: array_like
            Input array for the Matrix
        dtype: str or np.dtype, optional
            Optional datatype of the matrix

        Returns
        -------
        matrix: Matrix
        """
        obj = np.asarray(inputarr, dtype).view(cls)
        return obj

    @classmethod
    def zeros(cls, n, m=None, dtype=None):
        """ Initialize Matrix filled with zeros

        Parameters
        ----------
        n: int
            number of rows of the matrix
        m: int, optional
            number of collumns of the matrix. If not specified,
            matrix will be square (m=n)
        dtype: str or np.dtype, optional
            Optional datatype of the matrix

        Returns
        -------
        matrix: Matrix
        """
        m = n if m is None else m
        return cls(np.zeros((n, m)), dtype)

    @classmethod
    def nan(cls, n, m=None, dtype=None):
        """ Initialize Matrix filled with NaN

        Parameters
        ----------
        n: int
            number of rows of the matrix
        m: int, optional
            number of collumns of the matrix. If not specified,
            matrix will be square (m=n)
        dtype: str or np.dtype, optional
            Optional datatype of the matrix

        Returns
        -------
        matrix: Matrix
        """
        m = n if m is None else m
        return cls(np.full((n, m), np.nan), dtype)

    @classmethod
    def eye(cls, n, dtype=None):
        """ Initialize Matrix as unitary matrix

        Parameters
        ----------
        n: int
            size of the square matrix
        dtype: str or np.dtype, optional
            Optional datatype of the matrix

        Returns
        -------
        matrix: Matrix
        """
        return cls(np.eye(n), dtype)

    @classmethod
    def block(cls, arrays):
        """

        Parameters
        ----------
        arrays : array_like
            Array blocks to assemble. If all block shapes match the
            matrix blocks are configured

        Returns
        -------
        block_matrix: Matrix
        """
        n, m = len(arrays), len(arrays[0])
        sizes = [arrays[i][i].shape[0] for i in range(min(n, m))]
        for i in range(n):
            for j in range(m):
                if j > i:
                    if arrays[i][j] is None:
                        arrays[i][j] = np.zeros((sizes[i], sizes[j]))
                    if arrays[j][i] is None:
                        arrays[j][i] = np.zeros((sizes[j], sizes[i]))
        return cls(np.block(arrays))

    @classmethod
    def uniform(cls, n, m=None, low=0., high=1., dtype=None):
        """ Initialize Matrix filled with random values

        Parameters
        ----------
        n: int
            number of rows of the matrix
        m: int, optional
            number of collumns of the matrix. If not specified,
            matrix will be square (m=n)
        low: float, default: 0
            Lower boundary of the output interval
        high: float, default: 1
            Upper boundary of the output interval
        dtype: str or np.dtype, optional
            Optional datatype of the matrix

        Returns
        -------
        matrix: matrix
        """
        m = n if m is None else m
        return cls(np.random.uniform(low, high, (n, m)), dtype)

    def iter_indices(self, skip_diag=False):
        """ index generator of the Matrix

        Parameters
        ----------
        skip_diag: bool, optional
            if True, skip diagonal indices (where i == j), default: False

        Yields
        ------
        idx: tuple
            collumn- and row-indices of the matrix
        """
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if skip_diag and i == j:
                    continue
                yield i, j

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

    def equal(self, other):
        return np.array_equal(self, other)

    def almost_equal(self, other, thresh=1e-6):
        diff = np.abs(self - other)
        if np.any(diff > thresh):
            return False
        return True

    @property
    def h(self):
        return np.conj(self).T

    @property
    def is_hermitian(self):
        return self.almost_equal(self.h)

    @property
    def offdiag_numbers(self):
        return offdiag_numbers(self)

    @property
    def max_offdiag_number(self):
        return max_offdiag_number(self)

    def get_block_slices(self, size):
        return blockmatrix_slices(self.shape, size)

    # =========================================================================

    def inv(self):
        """ Matrix: Inverse of the Matrix """
        return Matrix(la.inv(self))

    def diag(self, offset=0):
        """ Get the optionally offset diagonal elements of the matrix

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
        """ Fill the optionally offset diagonal of the given array

        Parameters
        ----------
        val: scalar or array_like
            Value to be written on the diagonal, its type must be compatible
            with that of the array a.
        offset: int, default: 0
            offset index of diagonal. An offset of 1 results
            in the first upper offdiagonal of the matrix,
            an offset of -1 in the first lower offdiagonal.
        """
        fill_diagonal(self, val, offset)

    def eig(self):
        """ Calculate eigenvalues and -vectors of the matrix

        Returns
        -------
        eigenvalues: np.ndarray
            eigenvalues of the matrix
        eigenvectors: np.ndarray
            eigenvectors of the matrix
        """
        if self.is_hermitian:
            return la.eigh(self)
        else:
            return la.eig(self)

    def eigvals(self, num_range=None):
        """ np.ndarray: eigenvalues of the matrix """
        return la.eigvalsh(self, eigvals=num_range)

    def eigvecs(self):
        """ np.ndarray: eigenvectors of the matrix """
        return la.eig(self)[1]

    # =========================================================================

    def banded_matrix(self, lu=None):
        """ Convert the matrix to a banded matrix in the ordered diagonal form

        Parameters
        ----------
        lu: tuple of int, default: None
            Number of non-zero lower and upper diagonals. Value will be
            calculated if not specified.

        Returns
        -------
        ab: np.ndarray
            band matrix
        lu: tuple of int
            Number of non-zero lower and upper diagonals.
        """
        return ordered_diag_matrix(self, lu)

    def inv_banded(self, lu=None):
        """ Compute the (multiplicative) inverse using the banded matrix

        Parameters
        ----------
        lu: tuple of int, default: None
            Number of non-zero lower and upper diagonals. Value will be
            calculated if not specified.

        Returns
        -------
        ainv: np.ndarray
        """
        return inv_banded(self, lu)

    # =========================================================================

    def show(self, show=True, cmap=cc.m_coolwarm, norm_offset=0.2, colorbar=True, values=False,
             x_ticklabels=None, y_ticklabels=None, ticklabels=None, xrotation=45):
        """ Plot the matrix using the MatrixPlot object

        Parameters
        ----------
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
        mp.matshow(self)
        if values:
            mp.show_values()
        if colorbar:
            mp.show_colorbar()
        if max(self.shape) < 20:
            mp.set_tickstep(1)

        if ticklabels is not None:
            x_ticklabels = y_ticklabels = ticklabels
        mp.set_ticklabels(x_ticklabels, y_ticklabels, xrotation)
        if show:
            mp.show()
        return mp

    def __str__(self):
        x = max([len(str(self[i, j])) for i, j in self.iter_indices()])
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
