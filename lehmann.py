# -*- coding: utf-8 -*-
"""
Created on 17 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
import scipy.linalg as la
from scipy import integrate
from itertools import product
from scitools import Plot
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


class FockBasis:

    def __init__(self, n, spins=2):
        self.states = [FockState(idx, n) for idx in product(range(2**n), repeat=spins)]

    def __getitem__(self, item):
        return self.states[item]


class FockState:

    def __init__(self, ints, n=None):
        self.ints = list(ints)
        self.n = n

    def copy(self):
        return FockState(self.ints.copy(), self.n)

    def __repr__(self):
        string = ", ".join([binstr(x, self.n)[::-1] for x in self.ints])
        return f"State({string})"

    def __eq__(self, other):
        for x1, x2 in zip(self.ints, other.ints):
            if x1 != x2:
                return False
        return True

    def __getitem__(self, item):
        return self.ints[item]

    def __setitem__(self, item, value):
        self.ints[item] = value

    def array(self):
        return np.asarray([[int(b) for b in binstr(x, self.n)] for x in self.ints])

    @property
    def particles(self):
        return sum([bin(x)[2:].count("1") for x in self.ints])

    @property
    def spins(self):
        return [bin(x)[2:].count("1") for x in self.ints]

    @property
    def occupations(self):
        return np.sum(self.array(), axis=0)

    def create(self, i, spin):
        if get_bit(self[spin], i) == 1:
            return None
        new = self.copy()
        new[spin] = set_bit(new[spin], i, 1)
        return new

    def annihilate(self, i, spin):
        if get_bit(self.ints[spin], i) == 0:
            return None
        new = self.copy()
        new[spin] = set_bit(new[spin], i, 0)
        return new

    def hopping(self, other):
        n = 0
        for x1, x2 in zip(self.ints, other.ints):
            diff = x1 ^ x2
            if bin(diff)[2:].count("1") == 2:
                print(binstr(diff).index("1"))
                n += 1
        return n == 1


def main():
    n = 2
    basis = FockBasis(n)
    s1, s2 = basis[5], basis[6]
    print(s1, s2)
    print(s1.hopping(s2))


if __name__ == "__main__":
    main()
