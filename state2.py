# -*- coding: utf-8 -*-
"""
Created on 17 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from scipy.sparse import csr_matrix
from itertools import product
from sciutils import Binary
from cmpy import Operator, State2
from cmpy.models.siam import siam_operator, Siam

UP_CHAR = "\u2191"
DOWN_CHAR = "\u2193"
DOUBLE_CHAR = "d"
EMPTY_CHAR = "."
CHARS = {(1, 1): DOUBLE_CHAR, (0, 0): EMPTY_CHAR,
         (1, 0): UP_CHAR, (0, 1): DOWN_CHAR}


def basis_states(n=2, sort=False, key=None):
    idx = range(2 ** n)
    states = [State2(u, d, n) for u, d in product(idx, idx)]
    if sort:
        states = sorted(states, key=key)
    return states


def annihilation_operator(states, s, idx):
    n = len(states)
    row, col, data = list(), list(), list()
    flipper = states.index(s)
    for state in states:
        sbin = state.bin
        if int(sbin >> idx) & 1:
            row.append(int(sbin ^ flipper))
            col.append(int(sbin))
            data.append(state.phase(idx))
    return csr_matrix((data, (row, col)), shape=(n, n))


def operators(states, n):
    ops = list()
    idx = 0
    for i in range(n):
        # Up annihilation operator
        s = State2.single_up(i)
        op = annihilation_operator(states, s, idx)
        ops.append(op)
        idx += 1
        # Down annihilation operator
        s = State2.single_down(i)
        op = annihilation_operator(states, s, idx)
        ops.append(op)
        idx += 1
    return ops


class Basis:

    def __init__(self, n=2):
        self.n = n
        self.states = basis_states(self.n)
        self.ops = self._build_operators()

    def _build_operators(self):
        return [Operator(o) for o in operators(self.states, self.n)]

    def sort(self, *args, **kwargs):
        self.states = sorted(self.states, *args, **kwargs)
        self.ops = self._build_operators()


def main():
    u, eps_imp, eps_bath, v = 10, 0, 1, 2
    siam = Siam(u, eps_imp, eps_bath, v)
    ham = siam.ham_op.hamiltonian(u=5, eps_imp=1, eps_bath=2, v=1)
    print(ham)
    ham.show()


if __name__ == "__main__":
    main()
