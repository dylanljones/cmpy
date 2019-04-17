# -*- coding: utf-8 -*-
"""
Created on 25 Jan 2019
@author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from scipy import linalg as la
from .plotting import MatrixPlot
from .printing import format_num


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
        shape = array.shape
        if shape:
            i1, j1 = i + shape[0], j + shape[1]
            self[i:i1, j:j1] = array
        else:
            self[i, j] = array

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
        shape = array.shape
        if shape:
            i1, j1 = i + shape[0], j + shape[1]
            self[i:i1, j:j1] += array
        else:
            self[i, j] += array

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

        r0, rs = self.shape[0], block_size[0]
        c0, cs = self.shape[1], block_size[1]
        self.block_indices = np.moveaxis(np.mgrid[0:r0:rs, 0:c0:cs], 0, -1)
        self.block_size = block_size

    def reset_blocks(self):
        """ Reset blocks to None """
        self.block_indices = None
        self.block_size = None

    def _get_block_indices(self, i, j):
        """ Get the indices of block (i, j)

        Parameters
        ----------
        i: int
            row index of block
        j: int
            collumn index of block

        Returns
        -------
        start: tuple
            start indices of block
        stop:
            end indices of block
        """
        if self.block_indices is None:
            raise ValueError("Blocks are not configured yet!")
        r0, c0 = self.block_indices[i, j]
        r1, c1 = r0 + self.block_size[0], c0 + self.block_size[1]
        return (r0, c0), (r1, c1)

    def get_block(self, i, j):
        """ np.ndarray: Return block with block index (i, j)"""
        (r0, c0), (r1, c1) = self._get_block_indices(i, j)
        return self[r0:r1, c0:c1]

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
        (r0, c0), (r1, c1) = self._get_block_indices(i, j)
        self[r0:r1, c0:c1] = array

    # ==============================================================================================

    def inv(self):
        """ Matrix: Inverse of the Matrix """
        return Matrix(la.inv(self))

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
        diag_elements = np.diag(self)
        if not matrix:
            return diag_elements
        else:
            diag = Matrix.zeros(*self.shape, dtype=self.dtype)
            diag.fill_diag(diag_elements)
            return diag

    def off_diag(self):
        """ Matrix: get the non-diagonal matrix-elements """
        n = min(self.shape)
        off_diag = Matrix(self)
        off_diag.fill_diag(np.zeros(n))
        return off_diag

    def fill_diag(self, diag_elements):
        """ Fill the diagonal elements

        Parameters
        ----------
        diag_elements: scalar or array_like
            elements to be written on the diagonal,
        """
        np.fill_diagonal(self, diag_elements)

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

    # ==============================================================================================

    def show(self, show=True):
        """ Plot the matrix

        Parameters
        ----------
        show: bool, optional
            if True, call plt.show(), default: True
        """
        mp = MatrixPlot()
        mp.load(self)
        # Draw block lines

        if self.block_indices is not None:
            for r in [idx[0] for idx in self.block_indices[1:, 0]]:
                mp.line(row=r, color="0.6")

            for c in [idx[1] for idx in self.block_indices[0, 1:]]:
                mp.line(col=c, color="0.6")

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
                    s = str(np.real(val))
                elif np.real(val) == 0:
                    s = str(np.imag(val)) + "j"
                else:
                    s = str(val)
                line += f"{s:^{x}} "
            string += line[:-1] + "]\n"
        return string[:-1]


class Hamiltonian(Matrix):

    def __new__(cls, inputarr, num_orbitals=1, dtype=None):
        """ Initialize Hamiltonian for system with multiple sites and orbitals

        Parameters
        ----------
        inputarr: array_like
            Input array for the Hamiltonian
        num_orbitals: int, optional
            number of orbitals per site. The default is 1
        dtype: str or np.dtype, optional
            Optional datatype of the matrix

        Returns
        -------
        matrix: Matrix
        """
        inputarr = np.asarray(inputarr)
        obj = super().__new__(cls, inputarr, dtype)
        n = inputarr.shape[0]
        obj.n_sites = int(n / num_orbitals)
        obj.n_orbs = num_orbitals
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.n_sites = getattr(obj, 'n_sites', None)
        self.n_orbs = getattr(obj, 'n_orbs', None)

    @classmethod
    def zeros(cls, n_sites, n_orbitals=1, dtype=None):
        """ Initialize Hamiltonian filled with zeros for system with multiple sites and orbitals

        Parameters
        ----------
        n_sites: int
            number of rows of the matrix
        n_orbitals: int, optional
            number of orbitals per site. The default is 1
        dtype: str or np.dtype, optional
            Optional datatype of the matrix

        Returns
        -------
        matrix: matrix
        """
        n = n_sites * n_orbitals
        return cls(np.zeros((n, n)), n_orbitals, dtype)

    # ==============================================================================================

    @property
    def n(self):
        """int: size of the hamiltonain-axis 1 and 2"""
        return self.shape[0]

    def get(self, i_s, j_s):
        """ Get hamiltonian energy element (with all orbitals)

        Parameters
        ----------
        i_s: int
            index of first the site
        j_s: int
            index of second the site
        Returns
        -------
        energy: array_like
        """
        i, j = i_s*self.n_orbs, j_s*self.n_orbs
        return self[i:i+self.n_orbs, j:j+self.n_orbs]

    def set(self, i_s, j_s, array):
        """ Set hamiltonian energy element (with all orbitals)

        Parameters
        ----------
        i_s: int
            index of first the site
        j_s: int
            index of second the site
        array: array_like
            energy array for all orbitals
        """
        i, j = i_s*self.n_orbs, j_s*self.n_orbs
        self.insert(i, j, array)

    def set_energy(self, i, e):
        """ Set on-site energy of site

        Parameters
        ----------
        i: int
            index of site
        e: array_like or scalar
            energy array of site for all orbitals
        """
        e = np.asarray(e)
        self.set(i, i, e)

    def set_hopping(self, i, j, t):
        """ Set on-site energy of site

        Parameters
        ----------
        i: int
            index of first site
        j: int
            index of second site
        t: array_like or scalar
            hopping array between all orbitals of the two sites
        """
        i, j, = min(i, j), max(i, j)
        t = np.asarray(t)
        self.set(i, j, t)
        self.set(j, i, t.conj().T)

    # ==============================================================================================

    def ground_state(self):
        """ Get the eigenvalue and eigenvector of the ground state

        Returns
        -------
        eigval: float
        eigvec: np.ndarray
        """
        eigvals, eigvectors = self.eig()
        i = np.argmin(eigvals)
        return eigvals[i], eigvectors[i]

    def undressed(self):
        """ Get the undressed hamiltonian

        Returns
        -------
        ham: Hamiltonian
        """
        ham = self.copy()
        for i in range(self.n_sites):
            ham.set_energy(i, np.zeros((self.n_orbs, self.n_orbs)))
        return ham

    def greens(self, omega, only_diag=True):
        """ Calculate the greens function for the hamiltonian

        Parameters
        ----------
        omega: complex
            energy for the greens function
        only_diag: bool, optional
            only return diagonal elements of the greens function if True

        Returns
        -------
        greens: np.ndarray
        """
        omega = np.asarray(omega)
        # Calculate eigenvalues and -vectors of hamiltonian
        eigenvalues, eigenvectors = self.eig()
        eigenvectors_adj = np.conj(eigenvectors).T

        # Calculate greens-function
        subscript_str = "ij,...j,ji->...i" if only_diag else "ik,...k,kj->...ij"
        arg = np.subtract.outer(omega, eigenvalues)
        return np.einsum(subscript_str, eigenvectors_adj, 1 / arg, eigenvectors)

    def dos(self, omegas):
        """ Calculate the density of states

        Parameters
        ----------
        omegas: array_like or scalar
            energy values to calculate density of states

        Returns
        -------
        dos: np.ndarray
        """
        greens = self.greens(omegas, only_diag=True)
        dos = -1/np.pi * np.sum(greens.imag, axis=1)
        return dos

    # ==============================================================================================

    def show(self, show=True, ticklabels=None, show_blocks=False):
        """ Plot the Hamiltonian

        Parameters
        ----------
        show: bool, optional
            if True, call plt.show(), default: True
        ticklabels: array_like, optional
            Lables of the states of the hamiltonian
        show_blocks: bool, optional
            if True, show blocks of the orbitals, default is False
        """
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
