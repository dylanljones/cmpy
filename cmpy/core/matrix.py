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


class Matrix(np.ndarray):

    def __init__(self, *args, **kwargs):
        # in practice you probably will not need or want an __init__
        # method for your subclass
        self.block_indices = None
        self.block_sizes = None

    def __new__(cls, inputarr, dtype=None):
        obj = np.asarray(inputarr, dtype).view(cls)
        obj.block_indices = None
        obj.block_sizes = None
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.block_indices = getattr(obj, 'block_indices', None)
        self.block_sizes = getattr(obj, 'block_sizes', None)

    @classmethod
    def zeros(cls, n, m=None, dtype=None):
        m = n if m is None else m
        return cls(np.zeros((n, m)), dtype)

    @classmethod
    def eye(cls, n, dtype=None):
        return cls(np.eye(n), dtype)

    @property
    def is_blocked(self):
        return self.block_sizes is not None

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

    # ==============================================================================================

    def inv(self):
        return la.inv(self)

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
        """ np.ndarray: get the non-diagonal matrix-elements """
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

    def config_blocks(self, *block_sizes):
        n = len(block_sizes)
        sizes = list()
        for el in block_sizes:
            if hasattr(el, "__len__"):
                sizes.append(el)
            else:
                sizes.append([el, el])
        sizes = np.asarray(sizes)
        row_idx = [sum(sizes[:i, 0]) for i in range(n+1)]
        col_idx = [sum(sizes[:i, 1]) for i in range(n+1)]
        block_indices = np.zeros((n, n, 2, 2), dtype="int")
        for i in range(n):
            for j in range(n):
                block_indices[i, j, 0] = row_idx[i], col_idx[j]
                block_indices[i, j, 1] = row_idx[i+1], col_idx[j+1]
        self.block_indices = block_indices
        self.block_sizes = sizes

    def reset_blocks(self):
        self.block_indices = None
        self.block_sizes = None

    @property
    def block_shape(self):
        return self.block_indices.shape[:2]

    def config_uniform_blocks(self, block_size):
        n = min(self.shape)
        n_blocks = int(n / block_size)
        block_sizes = [[block_size, block_size]]*n_blocks
        self.config_blocks(*block_sizes)

    def get_block(self, i, j):
        (r0, c0), (r1, c1) = self.block_indices[i, j]
        return self[r0:r1, c0:c1]

    def set_block(self, i, j, array):
        (r0, c0), (r1, c1) = self.block_indices[i, j]
        self[r0:r1, c0:c1] = array

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
        if self.block_indices is not None:
            n = self.block_sizes.shape[0]
            row_idx = [sum(self.block_sizes[:i, 0]) for i in range(1, n)]
            col_idx = [sum(self.block_sizes[:i, 1]) for i in range(1, n)]
            for r in row_idx:
                mp.line(row=r, color="0.6")
            for c in col_idx:
                mp.line(col=c, color="0.6")
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
                    s = str(np.real(val))
                elif np.real(val) == 0:
                    s = str(np.imag(val)) + "j"

                else:
                    s = str(val)
                line += f"{s:^{x}} "
            string += line[:-1] + "]\n"
        return string[:-1]
