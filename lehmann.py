# -*- coding: utf-8 -*-
"""
Created on 17 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
import scipy.linalg as la
from scipy import integrate, sparse
from itertools import product
from scitools import Plot, Matrix
from cmpy import get_omegas
from cmpy.models import Siam


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


class BasisState:

    def __init__(self, spins, n_sites=None):
        if isinstance(spins, int):
            spins = [spins]
        self.spins = list(spins)
        self.n_sites = n_sites

    def copy(self):
        return self.__class__(self.spins.copy(), self.n)

    def __repr__(self):
        string = ", ".join([binstr(x, self.n_sites)[::-1] for x in self.spins])
        return f"State({string})"

    def __eq__(self, other):
        for x1, x2 in zip(self.spins, other.ints):
            if x1 != x2:
                return False
        return True

    def __getitem__(self, item):
        return self.spins[item]

    def __setitem__(self, item, value):
        self.spins[item] = value

    def array(self):
        return np.asarray([[int(b) for b in binstr(x, self.n_sites)] for x in self.spins])

    @property
    def n(self):
        return sum([bin(x)[2:].count("1") for x in self.spins])

    @property
    def s(self):
        return [bin(x)[2:].count("1") for x in self.spins]

    @property
    def occupations(self):
        return np.sum(self.array(), axis=0)

    @property
    def interactions(self):
        arr = self.occupations
        return (arr == 2).astype("int")

    def hopping(self, other):
        changed = 0
        hop_count = 0
        for s1, s2 in zip(self.spins, other.spins):
            diff = s1 ^ s2
            if diff:
                changed += 1
                hop = True
                indices = [i for i, char in enumerate(binstr(diff)) if char == "1"]
                if binstr(s1).count("1") != binstr(s2).count("1"):
                    hop = False
                elif len(indices) != 2:
                    hop = False
                else:
                    if abs(indices[0] - indices[1]) != 1:
                        hop = False
                if hop:
                    hop_count += 1
        return int(hop_count == 1) if changed == 1 else 0

    def create(self, i, spin):
        if get_bit(self[spin], i) == 1:
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


class FState(BasisState):

    EMPTY = r"."
    UP = u"\u2193"
    DOWN = u"\u2193"
    DOUBLE = r"d"

    def __init__(self, up, down=0, n_sites=2):
        super().__init__([up, down], n_sites)

    @property
    def label(self):
        chars = list()
        for i in range(self.n_sites):
            u = get_bit(self.spins[0], i)
            d = get_bit(self.spins[1], i)
            char = self.EMPTY
            if u and d:
                char = self.DOUBLE
            elif u:
                char = self.UP
            elif d:
                char = self.DOWN
            chars.append(char)
        return "".join(chars)

    def __str__(self):
        return self.label


class FBasis:

    def __init__(self, n, spins=2):
        self.states = [FState(*idx, n_sites=n) for idx in product(range(2**n), repeat=spins)]

    @property
    def n(self):
        return len(self.states)

    def sort(self, key):
        self.states.sort(key=key)

    def __getitem__(self, item):
        return self.states[item]

    def __iter__(self):
        return iter(self.states)


def hamiltonian(basis, u, eps, t):
    ham = Matrix.zeros(basis.n)
    for i in range(basis.n):
        s1 = basis[i]
        occ = np.sum(s1.occupations)
        inter = np.sum(s1.interactions)
        ham[i, i] = eps * occ + u * inter
        for j in range(basis.n):
            if i != j:
                s2 = basis[j]
                ham[i, j] = t * s1.hopping(s2)
    return ham


def main():
    u, eps, t = 4, 1, 2

    basis = FBasis(2, spins=1)
    basis.sort(key=lambda x: x.n)
    ham = hamiltonian(basis, u, eps, t)
    ham.show()




if __name__ == "__main__":
    main()
