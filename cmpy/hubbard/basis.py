# -*- coding: utf-8 -*-
"""
Created on 1 Dec 2018
@author: Dylan Jones

project: tightbinding
version: 1.0
"""
import numpy as np
from itertools import product


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

    @property
    def repr(self):
        parts = list()
        for i in range(self.n):
            string = self.UP_CHAR if self[0, i] else "."
            string += self.DN_CHAR if self[1, i] else "."
            parts.append(string)
        return " ".join(parts)

    def latex_string(self):
        parts = list()
        for i in range(self.n):
            string = r"$\uparrow$" if self[0, i] else "."
            string += r"$\downarrow$" if self[1, i] else "."
            parts.append(string)
        return " ".join(parts)

    def __repr__(self):
        return f"State({self.repr})"

    def __str__(self):
        parts = list()
        for i in range(self.n):
            string = self.UP_CHAR if self[0, i] else "."
            string += self.DN_CHAR if self[1, i] else "."
            parts.append(string)
        string = " ".join(parts)
        string += f" (N_e={self.num}, S={self.spin})"
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
