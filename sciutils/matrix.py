# -*- coding: utf-8 -*-
"""
Created on  06 2019
author: dylan

project: sciutils
version: 1.0
"""
import numpy as np
from scipy import linalg as la
from .plotting import MatrixPlot
from .terminal import format_num


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


def max_offdiag_number(mat):
    """ Calculate the maximal diagonal offset of the matrix

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
        lower, upper = max_offdiag_number(mat)

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
# MATRIX OBJECT
# =========================================================================


class Matrix(np.ndarray):

    def __init__(self, *args, **kwargs):
        # in practice you probably will not need or want an __init__
        # method for your subclass
        self.block_indices = None
        self.block_size = None

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
        obj.block_indices = None
        obj.block_size = None
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.block_indices = getattr(obj, 'block_indices', None)
        self.block_size = getattr(obj, 'block_size', None)

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
        matrix: matrix
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
        matrix: matrix
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
        matrix: matrix
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
        block_matrix
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
    def H(self):
        return np.conj(self).T

    @property
    def is_hermitian(self):
        return self.almost_equal(self.H)

    @property
    def max_offdiag_number(self):
        return max_offdiag_number(self)

    # =========================================================================

    @property
    def is_blocked(self):
        """ bool: True if blocks are configured """
        return self.block_size is not None

    @property
    def block_shape(self):
        """ tuple: shape of the blocked matrix """
        if self.is_blocked:
            return self.block_indices.shape[:2]
        else:
            return None

    def config_blocks(self, block_size):
        """ Configure the blocking of the Matrix

        Parameters
        ----------
        block_size: tuple or int
            row and column size of the block. If only a int is given
            block shape will be square
        """
        # Convert to tuple if int
        if not hasattr(block_size, "__len__"):
            block_size = (block_size, block_size)
        # Check size compability
        if (self.shape[0] % block_size[0] != 0) or (self.shape[1] % block_size[1] != 0):
            raise ValueError("Shape of Matrix must be divisible through block-size!")

        r0, rs = self.shape[0], block_size[0]
        c0, cs = self.shape[1], block_size[1]
        self.block_indices = np.moveaxis(np.mgrid[0:r0:rs, 0:c0:cs], 0, -1)
        self.block_size = block_size

    def reset_blocks(self):
        """ Reset blocks to None """
        self.block_indices = None
        self.block_size = None

    def block_idx(self, i, j):
        """ Get the indices of block (i, j)

        Parameters
        ----------
        i: int
            row index of block
        j: int
            collumn index of block

        Returns
        -------
        idx: tuple of slice
            row- and collumn index-slices of block
        """
        r, c = self.block_indices[i, j]
        r_idx = slice(r, r + self.block_size[0])
        c_idx = slice(c, c + self.block_size[1])
        return r_idx, c_idx

    def get_block(self, i, j):
        """ np.ndarray: Return block with block index (i, j)"""
        idx = self.block_idx(i, j)
        return self[idx]

    def set_block(self, i, j, array):
        """ Set block with block index (i, j)

        Parameters
        ----------
        i: int
            Row index of block
        j: int
            Collumns index of block
        array: array_like
            Data to fill
        """
        idx = self.block_idx(i, j)
        self[idx] = array

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

    def show(self, show=True, cmap="Greys", show_values=False):
        """ Plot the matrix using the MatrixPlot object

        Parameters
        ----------
        show: bool, optional
            if True, call plt.show(), default: True
        show_values: bool, default: False
            if True, print values in boxes
        cmap: str, default: "Greys"
            colormap used in the plot
        """
        mp = MatrixPlot(cmap=cmap)
        mp.load(self)
        mp.show_colorbar()
        if show_values:
            mp.show_values()
        if show:
            mp.show()
        return mp

    def print_mem(self):
        """ Print formatted string of memory usage"""
        print(format_num(self.nbytes, "b", 1024))

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
