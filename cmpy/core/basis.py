# -*- coding: utf-8 -*-
"""
Created on 1 Dec 2018
@author: Dylan Jones

project: tightbinding
version: 1.0
"""
import numpy as np
from itertools import product


class LatticeState(np.ndarray):

    EMPTY_CHAR = "."
    SINGE_CHAR = "x"
    DOWN_CHAR = "\u2193"
    UP_CHAR = "\u2191"
    UPDOWN_CHAR = "d"  # "\u21C5"

    def __new__(cls, up, down=None):
        arr = [up, down] if down is not None else [up]
        obj = np.asarray(arr, dtype="int").view(cls)
        return obj

    @classmethod
    def single(cls, n, i, spin=None):
        if spin is None:
            arr = np.zeros((1, n))
            idx = 0
        else:
            arr = np.zeros((2, n))
            idx = spin
        arr[idx, i] = 1
        return cls(*arr)

    @property
    def n(self):
        return self.shape[1]

    @property
    def spin(self):
        return self.shape[0] == 2

    @property
    def total_spin(self):
        return 1/2 * (np.sum(self[0]) - np.sum(self[1])) if self.spin else 0

    @property
    def num(self):
        return int(np.sum(self))

    @property
    def label(self):
        parts = list()
        for i in range(self.n):
            if self.spin:
                if self[0, i] and not self[1, i]:
                    string = self.UP_CHAR
                elif not self[0, i] and self[1, i]:
                    string = self.DOWN_CHAR
                elif self[0, i] and self[1, i]:
                    string = self.UPDOWN_CHAR
                else:
                    string = self.EMPTY_CHAR
            else:
                string = self.SINGE_CHAR if self[0, i] else self.EMPTY_CHAR
            parts.append(string)
        return " ".join(parts)

    def idx(self):
        idx = np.asarray(np.where(self > 0)).T
        return idx[:, 1]

    def equals(self, other):
        return np.all(self == other)

    def __repr__(self):
        return f"State({self.label})"

    def __str__(self):
        string = self.label
        string += f" (N_e={self.num}, S={self.total_spin})"
        return string


class LatticeBasis:

    def __init__(self, latt, spin=True, interacting=False):
        self.latt = None
        self.interacting = False
        self.all_states = list()
        self.states = list()
        self.spin = False

        self.initialize(latt, spin, interacting)

    @property
    def n(self):
        n = len(self.all_states)
        return n

    def initialize(self, latt, spin=True, interacting=False):
        n = latt.n
        if interacting:
            spin_states = list(product([0, 1], repeat=n))
            if spin:
                states = [LatticeState(up, down) for up, down in product(spin_states, repeat=2)]
            else:
                states = [LatticeState(x) for x in spin_states]
        else:
            states = list()
            if spin:
                for i in range(n):
                    states.append(LatticeState.single(n, i, 0))
                    states.append(LatticeState.single(n, i, 1))
            else:
                for i in range(n):
                    states.append(LatticeState.single(n, i))
        self.latt = latt
        self.interacting = interacting
        self.all_states = states
        self.states = states
        self.spin = spin

    def get_occupied(self, i):
        idx = self.states[i].idx()
        return idx

    def __getitem__(self, item):
        return self.states[item]

    def __str__(self):
        string = "Lattice Basis:"
        string += "\nSite " + " ".join([str(i+1) for i in range(self.latt.n)])
        for i in range(self.n):
            if i >= 100:
                string += "\n:"
                break
            string += f"\n{i+1:<5}{self.states[i].label} -> {list(self.get_occupied(i))}"
        return string
