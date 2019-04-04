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


class Basis:

    def __init__(self, *orbitals, spin=True):
        if spin:
            orbs = list()
            orbs += [orb + " up" for orb in orbitals]
            orbs += [orb + " down" for orb in orbitals]
        else:
            orbs = orbitals
        self.spin = spin
        self.orbs = orbs

        n = len(orbs)
        self.n = n
        self._eps = Matrix.zeros(n, dtype="complex")
        self._soc = Matrix.zeros(n, dtype="complex")
        self.hop = Matrix.zeros(n, dtype="complex")

    @property
    def eps(self):
        return self._eps + self._soc

    def find_orbit(self, orb):
        return [i for i in range(self.n) if self.orbs[i].startswith(orb)]

    def set_energy(self, orb, energy):
        for i in range(self.n):
            if self.orbs[i].startswith(orb):
                self.eps[i, i] = energy

    def set_hopping(self, orb1, orb2, hopping):
        indices_i = self.find_orbit(orb1)
        indices_j = self.find_orbit(orb2)
        for i, j in zip(indices_i, indices_j):
            self.hop[i, j] = hopping
            if i != j:
                self.hop[j, i] = hopping

    def set_soc(self, coupling=1.):
        h_soc = Matrix.zeros(self.n, dtype="complex")
        if self.spin is False:
            raise ValueError("SOC requires two different spin-types")
        for i, j in self.eps.iter_indices():
            orb1, s1 = self.orbs[i].split(" ")
            orb2, s2 = self.orbs[j].split(" ")
            h_soc[i, j] = get_soc(orb1, orb2, s1, s2)
        self._soc = h_soc * coupling

    def site_string(self, width=None):
        header = [orb.split(" ")[0] for orb in self.orbs]
        string = matrix_string(self.eps, width, col_header=header, row_header=header)
        return string

    def hop_string(self, width=None):
        header = [orb.split(" ")[0] for orb in self.orbs]
        string = matrix_string(self.hop, width, col_header=header, row_header=header)
        return string

    def show(self):
        header = list()
        for orb in self.orbs:
            if self.spin:
                orb, s = orb.split(" ")
                if s == "up":
                    orb += r"\uparrow"
                elif s == "down":
                    orb += r"\downarrow"
            header.append(f"${orb}$")

        plot1 = self.eps.show(False)
        plot1.ax.set_title("Site-hamiltonian")
        plot1.set_ticklabels(header, header)
        plot1.fig.tight_layout()
        plot2 = self.hop.show(False)
        plot2.fig.tight_layout()
        plot2.set_ticklabels(header, header)
        plot2.ax.set_title("Hopping-hamiltonian")
        plt.show()

    def __str__(self):
        string = "BASE:\n\n"
        string += "Site-hamiltonian:\n" + self.site_string() + "\n\n"
        string += "Hopping-hamiltonian:\n" + self.hop_string()
        return string


def s_basis(eps=0, t=1):
    b = Basis("s", spin=False)
    b.set_energy("s", eps)
    b.set_hopping("s", "s", t)
    return b


def sp3_basis(eps_s=0, eps_p=3, t_sss=-1, t_sps=0.75, t_pps=0.75, t_ppp=-0.25, soc=2):
    b = Basis("s", "p_x", "p_y", "p_z", spin=True)
    # Site energies
    b.set_energy("s", eps_s)
    b.set_energy("p", eps_p)
    # Hopping parameters
    b.set_hopping("s", "s", t_sss)
    b.set_hopping("s", "p_x", t_sps)
    b.set_hopping("p_x", "p_x", t_pps)
    b.set_hopping("p_y", "p_y", t_pps)
    b.set_hopping("p_z", "p_z", t_ppp)
    # spin orbit coupling
    b.set_soc(soc)
    return b
