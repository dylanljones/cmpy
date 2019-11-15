# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2019
author: Dylan Jones

project: dmft
version: 1.0
"""
import numpy as np
from itertools import product
from .operators import annihilation_operators


def basis_states(n):
    return list(range(int(n)))


def binstr(x, n=None):
    string = bin(x)[2:]
    n = n or len(string)
    return f"{string:0>{n}}"


def set_bit(binary, bit, value):
    if value:
        return binary | (1 << bit)
    else:
        return binary & ~(1 << bit)


def get_bit(binary, bit):
    return binary >> bit & 1


def check_filling(spins, particles=None):
    if particles is None:
        return True
    n = sum([bin(x)[2:].count("1") for x in spins])
    return n in particles


def check_spin(spins, spin_values=None):
    if spin_values is None:
        return True
    s = [bin(x)[2:].count("1") for x in spins]
    return 0.5 * (s[0] - s[1]) in spin_values


def check_sector(spins, n=None, s=None):
    return check_filling(spins, n) and check_spin(spins, s)


def iter_state_binaries(n_sites, n_spins, n=None, s=None):
    if n is not None and not hasattr(n, "__len__"):
        n = [n]
    if s is not None and not hasattr(s, "__len__"):
        s = [s]
    for spins in product(range(2 ** n_sites), repeat=n_spins):
        if check_sector(spins, n, s):
            yield spins


class FState:

    EMPTY = "."
    UP = "↑"
    DOWN = "↓"
    DOUBLE = "⇅"  # "d"

    def __init__(self, spins, n_sites=None):
        if isinstance(spins, int):
            spins = [spins]
        self.spins = list(spins)
        self.n_sites = n_sites

    def copy(self):
        return self.__class__(self.spins, self.n_sites)

    @property
    def label(self):
        chars = list()
        for i in range(self.n_sites):
            char = self.EMPTY
            u = get_bit(self.spins[0], i)
            if self.n_spins == 2:
                d = get_bit(self.spins[1], i)
                if u and d:
                    char = self.DOUBLE
                elif u:
                    char = self.UP
                elif d:
                    char = self.DOWN
            else:
                if u:
                    char = self.UP
            chars.append(char)
        return "".join(chars)[::-1]

    def __repr__(self):
        string = ", ".join([binstr(x, self.n_sites)[::-1] for x in self.spins])
        return f"State({string})"

    def __str__(self):
        return self.label

    def __eq__(self, other):
        return self.spins == other.spins

    def __getitem__(self, item):
        return self.spins[item]

    def __setitem__(self, item, value):
        self.spins[item] = value

    def array(self):
        return np.asarray([[int(b) for b in binstr(x, self.n_sites)] for x in self.spins])

    @property
    def n_spins(self):
        return len(self.spins)

    @property
    def n(self):
        return sum([bin(x)[2:].count("1") for x in self.spins])

    @property
    def s(self):
        num_spins = [bin(x)[2:].count("1") for x in self.spins]
        if self.n_spins == 2:
            return 0.5 * (num_spins[0] - num_spins[1])
        else:
            return 0.5 * sum(self.spins[0])

    @property
    def occupations(self):
        return np.sum(self.array(), axis=0)

    @property
    def interactions(self):
        arr = self.occupations
        return (arr == 2).astype("int")

    def hopping_indices(self, other):
        partners = list()
        for s1, s2 in zip(self.spins, other.spins):
            diff = s1 ^ s2
            if diff:
                if binstr(s1).count("1") == binstr(s2).count("1"):
                    indices = [i for i, char in enumerate(binstr(diff)) if char == "1"]
                    if len(indices) == 2:
                        partners.append(indices)
        return partners[0] if len(partners) == 1 else None

    def check_hopping(self, other, delta=None):
        indices = self.hopping_indices(other)
        if indices is None:
            return False
        elif delta is not None and abs(indices[1] - indices[0]) != delta:
            return False
        return True

    def create(self, i, spin):
        if get_bit(self.spins[spin], i) == 1:
            return None
        new = self.copy()
        new[spin] = set_bit(new[spin], i, 1)
        return new

    def annihilate(self, i, spin):
        if get_bit(self.spins[spin], i) == 0:
            return None
        new = self.copy()
        new[spin] = set_bit(new[spin], i, 0)
        return new


class FBasis:

    def __init__(self, n_sites, n_spins=2, states=None, n=None, s=None):
        self.n_sites = n_sites
        self.n_spins = n_spins
        if states is None:
            states = list()
            for spins in iter_state_binaries(n_sites, n_spins, n, s):
                states.append(FState(spins, n_sites))
        self.states = states

    @classmethod
    def free(cls, n_sites):
        states = [FState([2**i], n_sites) for i in range(n_sites)]
        return cls(n_sites, 1, states)

    @property
    def n(self):
        return len(self.states)

    @property
    def labels(self):
        return [s.label for s in self.states]

    def __str__(self):
        string = "Basis:"
        for s in self.states:
            string += f"\n  {s}"
        return string

    def __getitem__(self, item):
        return self.states[item]

    def __iter__(self):
        return iter(self.states)

    def __eq__(self, other):
        for s1, s2 in zip(self.states, other.states):
            if s1 == s2:
                return False
        return True

    def get(self, n=None, s=None):
        if n is not None and not hasattr(n, "__len__"):
            n = [n]
        if s is not None and not hasattr(s, "__len__"):
            s = [s]
        states = list()
        for state in self.states:
            if check_sector(state.spins, n, s):
                states.append(state)
        return states

    def subbasis(self, n=None, s=None):
        return self.__class__(self.n_sites, self.n_spins, self.get(n, s))

    def sort(self, key=None):
        if key is None:
            key=lambda s: s.n
        self.states.sort(key=key)

    def index(self, state):
        return self.states.index(state)

    def build_annihilation_ops(self):
        return annihilation_operators(self)
