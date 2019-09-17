# -*- coding: utf-8 -*-
"""
Created on 25 Jan 2019
@author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from sciutils import Matrix, eta
from .greens import gf_lehmann


class Hamiltonian(Matrix):

    @property
    def n(self):
        """int: size of the hamiltonain-axis 1 and 2"""
        return self.shape[0]

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

    def gf(self, omega=eta, mu=0., only_diag=True, banded=False):
        """ Calculate the greens function of the Hamiltonian

        Parameters
        ----------
        omega: complex or array_like, default: eta
            Energy e+eta of system (must be complex!). The default is zero (plus broadening).
        mu: float, default: 0
            Chemical potential of the system
        only_diag: bool, default: True
            only return diagonal elements of the greens function if True
        banded: bool, default: False
            Use the upper diagonal matrix for solving the eigenvalue problem. Full diagonalization is default

        Returns
        -------
        greens: np.ndarray
        """
        return gf_lehmann(self, omega, mu, only_diag, banded)

    def spectral(self, omega=eta, banded=False):
        """ Calculate the spectral function of the Hamiltonian

        Parameters
        ----------
        omega: complex or array_like, default: eta
            Energy e+eta of system (must be complex!). The default is zero (plus broadening).
        banded: bool, default: False
            Use the upper diagonal matrix for solving the eigenvalue problem. Full diagonalization is default

        Returns
        -------
        spec: np.ndarray
        """
        greens = gf_lehmann(self, omega, only_diag=True, banded=banded)
        return -np.sum(greens.imag, axis=1)

    def ldos(self, omega=eta, banded=False):
        """ Calculate the local density of states

        Parameters
        ----------
        omega: complex or array_like, default: eta
            Energy e+eta of system (must be complex!). The default is zero (plus broadening).
        banded: bool, default: False
            Use the upper diagonal matrix for solving the eigenvalue problem. Full diagonalization is default

        Returns
        -------
        ldos: np.ndarray
        """
        greens = self.gf(omega, only_diag=True, banded=banded)
        dos = -1 / np.pi * greens.imag
        return dos

    def dos(self, omega=eta, banded=False):
        """ Calculate the density of states

        Parameters
        ----------
        omega: complex or array_like, default: eta
            Energy e+eta of system (must be complex!). The default is zero (plus broadening).
        banded: bool, default: False
            Use the upper diagonal matrix for solving the eigenvalue problem. Full diagonalization is default

        Returns
        -------
        dos: np.ndarray
        """
        return np.sum(self.ldos(omega, banded), axis=1)

    def show(self, show=True, cmap=None, show_values=False, labels=None):
        """ Plot the Hamiltonian

        Parameters
        ----------
        show: bool, optional
            if True, call plt.show(), default: True
        cmap: str, optional
            colormap used in the plot
        show_values: bool, default: False
            if True, print values in boxes
        labels: list, optional
            Optional labels of the basis states of the matrix, default: None
        """
        mp = super().show(False, cmap=cmap, show_values=show_values, labels=labels)
        mp.tight()
        if show:
            mp.show()
        return mp


class HamiltonianCache:

    def __init__(self):
        self._ham = None

        self._memory = None
        self._mask = None

    def __bool__(self):
        return self._ham is not None

    def load(self, ham):
        self._ham = ham
        self._memory = ham.copy()
        mask = Matrix.zeros(ham.n, dtype="bool")
        mask.fill_diag(True)
        for i in range(1, ham.max_offdiag_number+1):
            mask.fill_diag(True, offset=i)
            mask.fill_diag(True, offset=-i)
        self._mask = mask

    def read(self):
        return self._ham

    def reset(self):
        self._ham[self._mask] = self._memory[self._mask]

    def clear(self):
        self._ham = None
        self._memory = None
        self._mask = None

    def __str__(self):
        string = "HamiltonianCache: "
        if not self:
            string += "empty"
        else:
            string += "\nHamiltonain:\n" + str(self._ham)
            string += "\nMemory:\n" + str(self._memory)
            string += "\nMask:\n" + str(self._mask)
        return string
