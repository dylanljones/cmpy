# -*- coding: utf-8 -*-
"""
Created on 29 Mar 2019
author: Dylan

project: cmpy2
version: 1.0
"""
import numpy as np
from itertools import product
from ..core import Lattice, Hamiltonian

# =========================================================================
# HUBBARD BASIS-STATES
# =========================================================================

class State:

    UP_CHAR = "\u2193"
    DN_CHAR = "\u2191"

    def __init__(self, up, down):
        self.arr = np.asarray((up, down))

    @property
    def n(self):
        return self.arr.shape[1]

    @property
    def spin(self):
        return 1/2 * (np.sum(self[0]) - np.sum(self[1]))

    @property
    def num(self):
        return int(np.sum(self.arr))

    @property
    def up(self):
        return self.arr[0]

    @property
    def dn(self):
        return self.arr[1]

    def site(self, i):
        return self.arr[:, i]

    def n_interaction(self):
        return sum([1 for i in range(self.n) if np.all(self[:, i])])

    def check_hopping(self, s):
        if self.num != s.num:
            return None
        if self.spin != s.spin:
            return None
        up_diff = np.where((self.up != s.up) > 0)[0]
        dn_diff = np.where((self.dn != s.dn) > 0)[0]
        hops = int((len(up_diff) + len(dn_diff)) / 2)
        if hops != 1:
            return None
        for diff in (up_diff, dn_diff):
            if diff.shape[0] > 0:
                return diff

    def __eq__(self, other):
        return np.all(self.arr == other.arr)

    def __getitem__(self, item):
        return self.arr[item]

    def _get_sitechar(self, i, latex=False):
        if self[0, i] and self[1, i]:
            return "d"
        elif self[0, i] and not self[1, i]:
            return self.UP_CHAR if not latex else r"$\uparrow$"
        elif not self[0, i] and self[1, i]:
            return self.DN_CHAR if not latex else r"$\downarrow$"
        return "."

    @property
    def repr(self):
        return " ".join([self._get_sitechar(i) for i in range(self.n)])

    def latex_string(self):
        return " ".join([self._get_sitechar(i, latex=True) for i in range(self.n)])

    def __repr__(self):
        return f"State({self.repr})"

    def __str__(self):
        string = self.repr + f" (N_e={self.num}, S={self.spin})"
        return string


class Basis:

    def __init__(self, n_sites):
        spin_states = list(product([0, 1], repeat=n_sites))
        self._all_states = [State(up, down) for up, down in product(spin_states, repeat=2)]

        self.n = 0
        self.s = 0
        self.states = list()

    @staticmethod
    def spin(up_state, down_state):
        return 1/2 * (sum(up_state) - sum(down_state))

    def state_strings(self):
        return [str(s) for s in self.states]

    def state_latex_strings(self):
        return [s.latex_string() for s in self.states]

    def get_states(self, n=None, spin=None):
        self.n = n
        self.s = spin
        sector_states = self._all_states
        for state in sector_states[:]:
            if (n is not None) and (state.num != n):
                sector_states.remove(state)
            elif (spin is not None) and (abs(state.spin) != spin):
                sector_states.remove(state)
        self.states = sector_states
        return sector_states


def build_hamiltonian(states, eps, t, u, mu=0):
    n = len(states)
    ham = Hamiltonian.zeros(n)
    t_conj = np.conj(t)

    for i in range(n):
        state = states[i]
        ham[i, i] =  eps + u * state.n_interaction()
        for j in range(n):
            if i < j:
                hops = state.check_hopping(states[j])
                if hops is not None:
                    i1, i2 = hops
                    if abs(i1 - i2) == 1:
                        ham[i, j] = t
                        ham[j, i] = t_conj
    return ham

# =========================================================================
# HUBBARD MODEL
# =========================================================================


class HubbardModel:

    def __init__(self, eps=0., t=1., u=10, mu=None):
        self.lattice = Lattice.square()
        self.basis = None
        self.states = list()

        self.eps = eps
        self.t = t
        self.u = u
        self.mu = self.u / 2 if mu is None else mu

        self.build(x=2)

    @property
    def state_labels(self):
        return [x.repr for x in self.states]

    def build(self, x=2, y=1):
        self.lattice.build((x, y))
        self.basis = Basis(int(x * y))

    def hamiltonian(self, n=None, spin=None):
        states = self.basis.get_states(n, spin)
        self.states = states
        return build_hamiltonian(states, self.eps, self.t, self.u, self.mu)

    def __repr__(self):
        eps = u"\u03b5".encode("utf-8")
        mu = u"\u03bc".encode()
        return f"Hubbard({eps}={self.eps}, t={self.t}, U={self.u}, {mu}={self.mu})"
