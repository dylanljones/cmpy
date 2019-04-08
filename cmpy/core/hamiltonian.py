# -*- coding: utf-8 -*-
"""
Created on 6 Dec 2018
@author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from .matrix import Matrix, SparseMatrix


def arr_shape(array):
    shape = array.shape
    return shape if len(shape) else (1, 1)


class Hamiltonian(Matrix):

    def __new__(cls, inputarr, num_orbitals=1, dtype=None):
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
        n = n_sites * n_orbitals
        return cls(np.zeros((n, n)), n_orbitals, dtype)

    @classmethod
    def eye(cls, n_sites, n_orbitals=1, dtype=None):
        return cls(np.eye(n_sites * n_orbitals), n_orbitals, dtype)

    # ==============================================================================================

    @property
    def n(self):
        return self.shape[0]

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

    def set(self, i_s, j_s, array):
        i, j = i_s*self.n_orbs, j_s*self.n_orbs
        self.insert(i, j, array)

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


class SparseHamiltonian(SparseMatrix):

    def __init__(self, inputarr=None, shape=None, n_orbitals=1, dtype=None):
        super().__init__(inputarr, shape, dtype)
        self.n_sites = None
        self.n_orbs = n_orbitals

        if inputarr is not None:
            self.n_sites = int(inputarr.shape[0]/n_orbs)

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
