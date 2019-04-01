# -*- coding: utf-8 -*-
"""
Created on 6 Dec 2018
@author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from .matrix import Matrix


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
        i0, j0 = i_s*self.n_orbs, j_s*self.n_orbs
        shape = arr_shape(array)
        i1, j1 = i0 + shape[0], j0 + shape[1]
        self[i0:i1, j0:j1] = array

    # ==============================================================================================

    def ground_state(self):
        eigvals, eigvectors = self.eig()
        i = np.argmin(eigvals)
        return eigvals[i], eigvectors[i]

    def undressed(self):
        ham = self.copy()
        ham.set_all_energies(np.zeros((self.n_orbs, self.n_orbs)))
        return ham

    def dos(self, omegas):
        greens = self.greens(omegas, only_diag=True)
        dos = -1/np.pi * np.sum(greens.imag, axis=1)
        return dos

    def greens(self, omega, only_diag=True):
        omega = np.asarray(omega)
        # Calculate eigenvalues and -vectors of hamiltonian
        eigenvectors_adj, eigenvalues, eigenvectors = self.eig()

        # Calculate greens-function
        subscript_str = "ij,...j,ji->...i" if only_diag else "ik,...k,kj->...ij"
        arg = np.subtract.outer(omega, eigenvalues)
        return np.einsum(subscript_str, eigenvectors_adj, 1 / arg, eigenvectors)

    # ==============================================================================================

    def show(self, show=True, ticklabels=None):
        mp = super().show(False)
        # if self.n_orbs > 1:
        #     row_idx = [i * self.n_orbs for i in range(1, self.n_sites)]
        #     col_idx = [i * self.n_orbs for i in range(1, self.n_sites)]
        #     for r in row_idx:
        #         mp.line(row=r, color="0.6")
        #     for c in col_idx:
        #         mp.line(col=c, color="0.6")
        if ticklabels is not None:
            mp.set_ticklabels(ticklabels, ticklabels)
        if show:
            mp.show()
        return mp
