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

    def show(self, show=True, ticklabels=None, cmap="Greys", show_values=False):
        """ Plot the Hamiltonian

        Parameters
        ----------
        show: bool, optional
            if True, call plt.show(), default: True
        ticklabels: array_like, optional
            Lables of the states of the hamiltonian
        cmap: str, default: "Greys"
            colormap used in the plot
        show_values: bool, default: False
            if True, print values in boxes
        show_blocks: bool, optional
            if True, show blocks of the orbitals, default is False
        """
        mp = super().show(False, cmap=cmap, show_values=show_values)
        if ticklabels is not None:
            mp.set_basislabels(ticklabels, ticklabels)
        mp.tight()
        if show:
            mp.show()
        return mp


class TbHamiltonian(Hamiltonian):

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
        """ Initialize Hamiltonian filled with zeros for a system with multiple sites and orbitals

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

    @classmethod
    def nan(cls, n_sites, n_orbitals=1, dtype=None):
        """ Initialize Hamiltonian filled with NaN's for a system with multiple sites and orbitals

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
        return cls(np.full((n, n), np.nan), n_orbitals, dtype)

    # ==============================================================================================

    @property
    def n(self):
        """int: size of the hamiltonain-axis 1 and 2"""
        return self.shape[0]

    @property
    def max_offdiag_number(self):
        return max(super().max_offdiag_number)

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
        if not array.shape:
            array = np.asarray([array])
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

    def show(self, show=True, ticklabels=None, cmap="Greys", show_values=False, show_blocks=True):
        """ Plot the Hamiltonian

        Parameters
        ----------
        show: bool, optional
            if True, call plt.show(), default: True
        ticklabels: array_like, optional
            Lables of the states of the hamiltonian
        cmap: str, default: "Greys"
            colormap used in the plot
        show_values: bool, default: False
            if True, print values in boxes
        show_blocks: bool, optional
            if True, show blocks of the orbitals, default is False
        """
        mp = super().show(False, cmap=cmap, show_values=show_values)
        if show_blocks and self.n_orbs > 1:
            row_idx = [i * self.n_orbs for i in range(1, self.n_sites)]
            col_idx = [i * self.n_orbs for i in range(1, self.n_sites)]
            for r in row_idx:
                mp.line(row=r, color="r", lw=2)
            for c in col_idx:
                mp.line(col=c, color="0.6")
        if ticklabels is not None:
            mp.set_ticklabels(ticklabels, ticklabels)
        mp.tight()
        if show:
            mp.show()
        return mp
