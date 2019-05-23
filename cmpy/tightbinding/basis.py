# -*- coding: utf-8 -*-
"""
Created on 1 Dec 2018
@author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
import matplotlib.pyplot as plt
from cmpy.core import Matrix, matrix_string

SPINS = "up", "down"
ORBS = "p_x", "p_y", "p_z"

s_0 = np.zeros((2, 2))
# Pauli matrices
s_x = np.array([[0, 1], [1, 0]])
s_y = np.array([[0, -1j], [1j, 0]])
s_z = np.array([[1, 0], [0, -1]])

SOC = np.array([[s_0,     -1j*s_z, +1j*s_y],
                [+1j*s_z,     s_0, -1j*s_x],
                [-1j*s_y, +1j*s_x,     s_0]])


def get_soc(orb1, orb2, s1, s2):
    if isinstance(orb1, str) and isinstance(orb2, str):
        if orb1 == "s" or orb2 == "s":
            return 0
        orb1, orb2 = ORBS.index(orb1), ORBS.index(orb2)
    if isinstance(s1, str) and isinstance(s2, str):
        s1, s2 = SPINS.index(s1), SPINS.index(s2)
    s = SOC[orb1, orb2]
    return s[s1, s2]

class State:

    def __init__(self, orb, spin=None):
        self.orb = orb
        self.spin = spin

    def is_orbit(self, orb):
        return self.orb == orb

    def is_spin(self, spin):
        return self.spin == spin

    def is_state(self, orb, spin):
        return self.is_orbit(orb) and self.is_spin(spin)

    def __str__(self):
        return f"{self.orb} {self.spin}"

    def __repr__(self):
        return f"State({self.orb}, {self.spin})"


class Basis:

    def __init__(self, *orbitals, spin=True, ordering="spin"):
        self.soc = None
        self.spin = spin
        if spin:
            states = [State(orb, "up") for orb in orbitals]
            states += [State(orb, "down") for orb in orbitals]
        else:
            states = [State(orb) for orb in orbitals]
        self.states = states
        self.sort_states(ordering)

        n = len(self.states)
        self.n = n
        self._eps = np.zeros((n, n), dtype="complex")
        self._soc = np.zeros((n, n), dtype="complex")
        self._hop = np.zeros((n, n), dtype="complex")

    def sort_states(self, mode="spin"):
        if mode == "spin":
            states = self.states
            states.sort(key=lambda s: s.spin)
            self.states = states[::-1]
        elif mode == "orb":
            states = self.states
            states.sort(key=lambda s: s.orb)
            self.states = states
        else:
            raise ValueError(f"ERROR: Sort-mode {mode} not recognized (Allowed modes: spin, orb)")

    @property
    def eps(self):
        """ np.ndarray: on-site energies of all orbitals"""
        return self._eps + self._soc

    @property
    def hop(self):
        """ np.ndarray: hopping energies of all orbitals"""
        return self._hop

    def find(self, txt):
        """ Find indices of orbitals that match query

        Parameters
        ----------
        txt: str
            search string

        Returns
        -------
        indices: list
        """
        return [i for i in range(self.n) if str(self.states[i]).startswith(txt)]

    def set_energy(self, orb, energy):
        """ Set on-site energy for a orbital

        Parameters
        ----------
        orb: str
            name of orbital
        energy: float
            energy value of orbital
        """
        for i in range(self.n):
            if self.states[i].orb.startswith(orb):
                self._eps[i, i] = energy

    def set_hopping(self, orb1, orb2, hopping):
        """ Set hopping energy between two orbitals

        Parameters
        ----------
        orb1: str
            name of first orbital
        orb2: str
            name of second orbital
        hopping: float
            hopping energy between the orbitals
        """
        indices_i = self.find(orb1)
        indices_j = self.find(orb2)
        for i, j in zip(indices_i, indices_j):
            self.hop[i, j] = hopping
            if i != j:
                self.hop[j, i] = hopping

    def set_soc(self, coupling=1.):
        """ Set the spin-orbit-coupling strength

        Parameters
        ----------
        coupling: float
            coupling strength, default: 1.
        """
        n = self.n
        self.soc = coupling
        h_soc = np.zeros((n, n), dtype="complex")
        if self.spin is False:
            raise ValueError("SOC requires two different spin-types")
        for i in range(n):
            for j in range(n):
                state1, state2 = self.states[i], self.states[j]
                orb1, s1 = state1.orb, state1.spin
                orb2, s2 = state2.orb, state2.spin
                h_soc[i, j] = get_soc(orb1, orb2, s1, s2)
        self._soc = h_soc * coupling

    # =========================================================================

    def labels(self):
        labels = list()
        for state in self.states:
            if self.spin:
                orb, s = state.orb, state.spin
                if s == "up":
                    orb += r"\uparrow"
                elif s == "down":
                    orb += r"\downarrow"
            labels.append(f"${orb}$")
        return labels

    def show(self):
        labels = self.labels()
        eps = Matrix(self.eps)
        plot1 = eps.show(False)
        plot1.ax.set_title("Site-hamiltonian")
        plot1.set_ticklabels(labels, labels)
        plot1.tight()

        hop = Matrix(self.hop)
        plot2 = hop.show(False)
        plot2.fig.tight_layout()
        plot2.set_ticklabels(labels, labels)
        plot2.ax.set_title("Hopping-hamiltonian")
        plot2.tight()
        plt.show()

    def site_string(self, width=None):
        header = [s.orb for s in self.states]
        string = matrix_string(self.eps, width, col_header=header, row_header=header)
        return string

    def hop_string(self, width=None):
        header = [s.orb for s in self.states]
        string = matrix_string(self.hop, width, col_header=header, row_header=header)
        return string

    def __str__(self):
        string = "BASE:\n\n"
        string += "Site-hamiltonian:\n" + self.site_string() + "\n\n"
        string += "Hopping-hamiltonian:\n" + self.hop_string()
        return string


def s_basis(eps=0, t=1, spin=False):
    """ Basis object for a system with s orbitals

    Parameters
    ----------
    eps: float, optional
        on-site energy of s orbital, default: 0.
    t: float, optional
        hopping energy between the s orbitals, default: 1.
    spin: bool, default: True
        Flag if both spin channels should be used

    Returns
    -------
    basis: Basis
    """
    b = Basis("s", spin=spin)
    b.set_energy("s", eps)
    b.set_hopping("s", "s", t)
    return b


def p3_basis_2c(eps_p=0, t_pps=1, t_ppp=1, d=None, spin=True, soc=0, ordering="spin"):
    """ Two-center basis object for a system with p_x, p_y, and p_z orbitals

    Parameters
    ----------
    eps_p: float, optional, default: 0.
        on-site energy of p orbital
    t_pps: float, default: 0.75
        symmetric hopping energy between the p orbitals
    t_ppp: float, default: -0.25
        antisymmetric hopping energy between the p orbitals
    d: array_like, default: None
        direction vector of hopping
    spin: bool, default: True
        Flag if both spin channels should be used
    soc: float, default: 1
        Spin-orbit-coupling strength
    ordering: string, default: "spin"
        ordering parameter for states

    Returns
    -------
    basis: Basis
    """
    d = np.ones(3) if d is None else d
    dx, dy, dz = d

    basis = Basis("p_x", "p_y", "p_z", ordering=ordering, spin=spin)
    basis.set_energy("p", eps_p)
    basis.set_hopping("p_x", "p_x", dx ** 2 + (1 - dx ** 2) * t_ppp)
    basis.set_hopping("p_y", "p_y", dy ** 2 + (1 - dy ** 2) * t_ppp)
    basis.set_hopping("p_z", "p_z", dz ** 2 + (1 - dz ** 2) * t_ppp)
    basis.set_hopping("p_x", "p_y", dx * dy * (t_pps - t_ppp))
    basis.set_hopping("p_y", "p_z", dy * dz * (t_pps - t_ppp))
    basis.set_hopping("p_z", "p_x", dz * dx * (t_pps - t_ppp))
    if spin and soc:
        basis.set_soc(soc)
    return basis


def p3_basis(eps_p=0, t_pps=1, t_ppp=1, d=None, spin=True, soc=0, ordering="spin"):
    return p3_basis_2c(eps_p, t_pps, t_ppp, d, spin, soc, ordering)


def sp3_basis_2c(eps_s=0,  eps_p=0, t_sss=-1, t_sps=0.75, t_pps=0.75, t_ppp=-0.25, d=None,
                 spin=True, soc=0, ordering="spin"):
    """ Two-center basis object for a system with p_x, p_y, and p_z orbitals

    Parameters
    ----------

    eps_s: float, optional, default: 0.
        on-site energy of p orbital
    eps_p: float, optional, default: 3.
        on-site energy of p orbital
    t_sss: float, default: -1.
        symmetric hopping energy between the s orbitals
    t_sps: float, default: 0.75
        symmetric hopping energy between the s and p orbitals
    t_pps: float, default: 0.75
        symmetric hopping energy between the p orbitals
    t_ppp: float, default: -0.25
        antisymmetric hopping energy between the p orbitals
    d: array_like, default: None
        direction vector of hopping
    spin: bool, default: True
        Flag if both spin channels should be used
    soc: float, default: 1
        Spin-orbit-coupling strength
    ordering: string, default: "spin"
        ordering parameter for states

    Returns
    -------
    basis: Basis
    """
    d = np.ones(3) if d is None else d
    dx, dy, dz = d

    basis = Basis("s", "p_x", "p_y", "p_z", ordering=ordering, spin=spin)
    basis.set_energy("s", eps_s)
    basis.set_energy("p", eps_p)
    basis.set_hopping("s", "s", t_sss)
    basis.set_hopping("s", "p_x", dx * t_sps)
    basis.set_hopping("s", "p_x", dy * t_sps)
    basis.set_hopping("s", "p_x", dz * t_sps)
    basis.set_hopping("p_x", "p_x", dx ** 2 + (1 - dx ** 2) * t_ppp)
    basis.set_hopping("p_y", "p_y", dy ** 2 + (1 - dy ** 2) * t_ppp)
    basis.set_hopping("p_z", "p_z", dz ** 2 + (1 - dz ** 2) * t_ppp)
    basis.set_hopping("p_x", "p_y", dx * dy * (t_pps - t_ppp))
    basis.set_hopping("p_y", "p_z", dy * dz * (t_pps - t_ppp))
    basis.set_hopping("p_z", "p_x", dz * dx * (t_pps - t_ppp))

    if spin and soc:
        basis.set_soc(soc)
    return basis


def sp3_basis(eps_s=0,  eps_p=0, v_sss=-1, v_sps=0.75, v_pps=0.75, v_ppp=-0.25, d=None,
              spin=True, soc=0, ordering="spin"):
    return sp3_basis_2c(eps_s,  eps_p, v_sss, v_sps, v_pps, v_ppp, d, spin, soc, ordering)

#
#
# def p3_basis(eps_p=0, t_pps=0.75, t_ppp=-0.25, soc=1., ordering="spin"):
#     """ Basis object for a system with p_x, p_y, and p_z orbitals
#
#     Parameters
#     ----------
#     eps_p: float, optional
#         on-site energy of p orbital, default: 0.
#     t_pps: float, optional
#         symmetric hopping energy between the p orbitals, default: 0.75
#     t_ppp: float, optional
#         antisymmetric hopping energy between the p orbitals, default: -0.25
#     soc: float, optional
#         Spin-orbit-coupling strength, default: 1.
#
#     Returns
#     -------
#     basis: Basis
#     """
#     b = Basis("p_x", "p_y", "p_z", spin=True, ordering=ordering)
#     # Site energies
#     b.set_energy("p", eps_p)
#     # Hopping parameters
#     b.set_hopping("p_x", "p_x", t_pps)
#     b.set_hopping("p_y", "p_y", t_pps)
#     b.set_hopping("p_z", "p_z", t_ppp)
#     # spin orbit coupling
#     b.set_soc(soc)
#     return b
#
#
# def sp3_basis(eps_s=0, eps_p=3, t_sss=-1., t_sps=0.75, t_pps=0.75, t_ppp=-0.25,
#               soc=1., ordering="spin"):
#     """ Basis object for a system with s, p_x, p_y, and p_z orbitals
#
#     Parameters
#     ----------
#     eps_s: float, optional
#         on-site energy of s orbital, default: 0.
#     eps_p: float, optional
#         on-site energy of p orbital, default: 3.
#     t_sss: float, optional
#         symmetric hopping energy between the s orbitals, default: -1.
#     t_sps: float, optional
#         symmetric hopping energy between the s and p orbitals, default: 0.75
#     t_pps: float, optional
#         symmetric hopping energy between the p orbitals, default: 0.75
#     t_ppp: float, optional
#         antisymmetric hopping energy between the p orbitals, default: -0.25
#     soc: float, optional
#         Spin-orbit-coupling strength, default: 1.
#
#     Returns
#     -------
#     basis: Basis
#     """
#     b = Basis("s", "p_x", "p_y", "p_z", spin=True, ordering=ordering)
#     # Site energies
#     b.set_energy("s", eps_s)
#     b.set_energy("p", eps_p)
#     # Hopping parameters
#     b.set_hopping("s", "s", t_sss)
#     b.set_hopping("s", "p_x", t_sps)
#     b.set_hopping("p_x", "p_x", t_pps)
#     b.set_hopping("p_y", "p_y", t_pps)
#     b.set_hopping("p_z", "p_z", t_ppp)
#     # spin orbit coupling
#     b.set_soc(soc)
#     return b
