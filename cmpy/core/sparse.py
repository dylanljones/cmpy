# -*- coding: utf-8 -*-
"""
Created on 6 Dec 2018
@author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as la_s
from scipy import linalg as la
from .matrix import Matrix


class SparseMatrix:

    def __init__(self, inputarr=None, shape=None, dtype=None):
        self.shape = shape
        self.dtype = dtype
        self.data = dict()

        self.block_size = None
        self.block_indices = None

        if shape is None and inputarr is not None:
            self.shape = inputarr.shape
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    self.__setitem__((i, j), inputarr[i, j])

    @classmethod
    def zeros(cls, n, m=None, dtype=None):
        m = n if m is None else m
        return cls(shape=(n, m), dtype=dtype)

    @classmethod
    def coo_matrix(cls, data, shape, dtype=None):
        self = cls.zeros(*shape, dtype=dtype)
        self.data = data
        return self

    @property
    def n_elements(self):
        return len(list(self.data.keys()))

    @property
    def density(self):
        return self.n_elements / (np.prod(self.shape))

    def _assert_index_in_range(self, idx, axis):
        if (idx < 0) or (idx >= self.shape[axis]):
            msg = f"Index {idx} is out of bounds for axis 0 with size {self.shape[axis]}"
            raise IndexError(msg)

    def __getitem__(self, item):
        self._assert_index_in_range(item[0], 0)
        self._assert_index_in_range(item[1], 1)
        return self.data.get(item, 0)

    def __setitem__(self, item, value):
        if not value:
            if item in self.data:
                del self.data[item]
            return
        self._assert_index_in_range(item[0], 0)
        self._assert_index_in_range(item[1], 1)
        self.data.update({item: value})

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
        indices = self._flatten_indices(i, j, array.shape)
        array = array.flatten()
        for idx in range(indices.shape[0]):
            i, j = indices[idx]
            self.__setitem__((i, j), array[idx])

    def add(self, i, j, array):
        indices = self._flatten_indices(i, j, array.shape)
        array = array.flatten()
        for idx in range(indices.shape[0]):
            _i, _j = indices[idx]
            val = self.__getitem__((_i, _j)) + array[idx]
            self.__setitem__((_i, _j), val)

    # ==============================================================================================

    def toarray(self):
        arr = np.zeros(self.shape, dtype=self.dtype)
        for idx, value in self.data.items():
            i, j = idx
            arr[i, j] = value
        return arr

    def tomatrix(self):
        m = Matrix(self.toarray())
        if self.is_blocked:
            m.block_indices = self.block_indices
            m.block_size = self.block_size
        return m

    def coo_data(self):
        rows, cols = list(), list()
        data = list()
        for idx, value in self.data.items():
            rows.append(idx[0])
            cols.append(idx[1])
            data.append(value)
        return rows, cols, data

    def scipy_coo(self):
        rows, cols, data = self.coo_data()
        return sparse.coo_matrix((data, (rows, cols)), shape=self.shape)

    # ==============================================================================================

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
            row and columns size of the block. If only a int is given
            block shape will be square
        """
        # Convert to tuple if int
        if not hasattr(block_size, "__len__"):
            block_size = (block_size, block_size)

        # Check size compability
        if (self.shape[0] % block_size[0] != 0) or (self.shape[1] % block_size[1] != 0):
            raise ValueError("Shape of Matrix must be divisible through block-size!")

        # Set block-indices and size
        rows = np.arange(0, self.shape[0] + 1, block_size[0])
        cols = np.arange(0, self.shape[1] + 1, block_size[1])
        n, m = rows.shape[0] - 1, cols.shape[0] - 1
        block_indices = np.zeros((n, m, 2, 2), dtype="int")
        for i in range(n):
            for j in range(m):
                block_indices[i, j, 0] = rows[i], cols[j]
                block_indices[i, j, 1] = rows[i+1], cols[j+1]

        self.block_size = block_size
        self.block_indices = block_indices

    def _assert_blocks(self):
        if self.block_indices is None:
            raise ValueError("Blocks are not configured yet!")

    def reset_blocks(self):
        """ Reset blocks to None """
        self.block_indices = None
        self.block_size = None

    @staticmethod
    def _flatten_indices(i, j, shape):
        if shape:
            indices = np.indices(shape).reshape(2, np.prod(shape))
            indices += np.array([i, j])[:, np.newaxis]
            return indices.T
        else:
            return np.asarray([[i, j]])

    def get_block(self, i, j):
        """ np.ndarray: Return block with block index i, j"""
        self._assert_blocks()
        (r0, c0), _ = self.block_indices[i, j]
        array = np.zeros(self.block_size, dtype=self.dtype)
        indices = self._flatten_indices(r0, c0, self.block_size)
        for idx in range(indices.shape[0]):
            i, j = indices[idx]
            array[i-r0, j-c0] = self.__getitem__((i, j))
        return array

    def set_block(self, i, j, array):
        """ Set block with block index i, j

        Parameters
        ----------
        i: int
            Row index of block
        j: int
            Collumns index of block
        array: array_like
            Data to fill
        """
        self._assert_blocks()
        (r0, c0), _ = self.block_indices[i, j]

        if not array.shape:
            array = np.array([[array]])

        indices = self._flatten_indices(r0, c0, self.block_size)
        array = array.flatten()
        for idx in range(indices.shape[0]):
            i, j = indices[idx]
            self.__setitem__((i, j), array[idx])

    # ==============================================================================================

    def inv(self):
        """ Matrix: Inverse of the Matrix """
        return la_s.inv(self.scipy_coo())

    def diag(self, matrix=False):
        """ Get the diagonal matrix-elements

        Parameters
        ----------
        matrix: bool, optional
            if true, return diagonal-elements as matrix

        Returns
        -------
        np.ndarray
        """
        n = min(self.shape)
        if not matrix:
            diag = np.zeros(n, self.dtype)
            for i in range(n):
                diag[i] = self.__getitem__((i, i))
        else:
            diag = SparseMatrix.zeros(*self.shape, dtype=self.dtype)
            for i in range(n):
                diag[i, i] = self.__getitem__((i, i))
        return diag

    def off_diag(self):
        """ Matrix: get the non-diagonal matrix-elements """
        off_diag = SparseMatrix.coo_matrix(self.data, self.shape, dtype=self.dtype)
        off_diag.fill_diag(0)
        return off_diag

    def fill_diag(self, diag_elements):
        """ Fill the diagonal elements

        Parameters
        ----------
        diag_elements: scalar or array_like
            elements to be written on the diagonal,
        """
        n = min(self.shape)
        if not hasattr(diag_elements, "__len__"):
            diag_elements = [diag_elements] * n
        for i in range(n):
            self[i, i] = diag_elements[i]

    def eig(self):
        """ Calculate eigenvalues and -vectors of the matrix

        Returns
        -------
        eigenvalues: np.ndarray
            eigenvalues of the matrix
        eigenvectors: np.ndarray
            eigenvectors of the matrix
        """
        return la.eig(self.toarray())

    def eigvals(self, num_range=None):
        """ np.ndarray: eigenvalues of the matrix """
        return la.eigvalsh(self.toarray(), eigvals=num_range)

    # ==============================================================================================

    def __repr__(self):
        return f"SparseMatrix({self.shape}), {self.n_elements} elements)"

    def show(self, show=True):
        mat = self.tomatrix()
        return mat.show(show)

    def __str__(self):
        return str(self.toarray())


class SparseHamiltonian(SparseMatrix):

    def __init__(self, inputarr=None, shape=None, n_orbitals=1, dtype=None):
        super().__init__(inputarr, shape, dtype)
        self.n_sites = None
        self.n_orbs = n_orbitals

        if inputarr is not None:
            self.n_sites = int(inputarr.shape[0]/n_orbitals)

    @classmethod
    def zeros(cls, n_sites, n_orbitals=1, dtype=None):
        n = n_sites * n_orbitals
        self = cls(shape=(n, n), dtype=dtype)
        self.n_sites = n_sites
        self.n_orbs = n_orbitals
        return self

    # ==============================================================================================

    @property
    def n(self):
        return self.shape[0]

    def set(self, i, j, array):
        self.insert(i*self.n_orbs, j*self.n_orbs, array)

    def set_energy(self, i, e):
        e = np.asarray(e)
        self.set(i, i, e)

    def set_all_energies(self, e):
        for i in range(self.n_sites):
            self.set_energy(i, e)

    def set_hopping(self, i, j, t):
        i, j, = min(i, j), max(i, j)
        t = np.asarray(t)
        self.set(i, j, t)
        self.set(j, i, t.conj().T)

    # ==============================================================================================

    def ground_state(self):
        eigvals, eigvectors = self.eig()
        i = np.argmin(eigvals)
        return eigvals[i], eigvectors[i]

    def undressed(self):
        ham = self.copy()
        ham.set_all_energies(np.zeros((self.n_orbs, self.n_orbs)))
        return ham

    def greens(self, omega, only_diag=True):
        omega = np.asarray(omega)
        # Calculate eigenvalues and -vectors of hamiltonian
        eigenvalues, eigenvectors = self.eig()
        eigenvectors_adj = np.conj(eigenvectors).T

        # Calculate greens-function
        subscript_str = "ij,...j,ji->...i" if only_diag else "ik,...k,kj->...ij"
        arg = np.subtract.outer(omega, eigenvalues)
        return np.einsum(subscript_str, eigenvectors_adj, 1 / arg, eigenvectors)

    def dos(self, omegas):
        greens = self.greens(omegas, only_diag=True)
        dos = -1/np.pi * np.sum(greens.imag, axis=1)
        return dos

    # ==============================================================================================

    def show(self, show=True, ticklabels=None, show_blocks=False):
        mp = super().show(False)
        if show_blocks and self.n_orbs > 1:
            row_idx = [i * self.n_orbs for i in range(1, self.n_sites)]
            col_idx = [i * self.n_orbs for i in range(1, self.n_sites)]
            for r in row_idx:
                mp.line(row=r, color="0.6")
            for c in col_idx:
                mp.line(col=c, color="0.6")
        if ticklabels is not None:
            mp.set_ticklabels(ticklabels, ticklabels)
        if show:
            mp.show()
        return mp
