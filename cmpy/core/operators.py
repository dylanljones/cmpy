# -*- coding: utf-8 -*-
"""
Created on 18 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from scipy.sparse import csr_matrix
from .hamiltonian import Hamiltonian


def annihilate(state, i):
    if not int(state >> i) & 1:
        return None
    return state ^ (1 << i)


def create(state, i):
    if int(state >> i) & 1:
        return None
    return state ^ (1 << i)


def phase(state, i):
    particles = (state >> i + 1).particles
    return 1 if particles % 2 == 0 else -1


def annihilation_operator(states, idx):
    n = len(states)
    row, col, data = list(), list(), list()
    for j in range(n):
        state = states[j]
        other = annihilate(state, idx)
        if other is not None:
            try:
                i = states.index(other)
                val = phase(state, idx)
                row.append(i)
                col.append(j)
                data.append(val)
            except ValueError:
                pass
    return csr_matrix((data, (row, col)), shape=(n, n), dtype="int")


def annihilators(states, n):
    return [Operator.annihilation_operator(states, x) for x in range(n)]


class Operator:

    def __init__(self, array=None):
        if isinstance(array, Operator):
            array = array.csr
        self.csr = csr_matrix(array)

    @classmethod
    def annihilation_operator(cls, states, idx):
        mat = annihilation_operator(states, idx)
        return cls(mat)

    @classmethod
    def creation_operator(cls, idx, states):
        return cls.annihilation_operator(idx, states).dag

    def todense(self):
        return self.csr.todense()

    @property
    def dense(self):
        return self.csr.todense()

    @property
    def T(self):
        return Operator(self.csr.T)

    @property
    def dag(self):
        return Operator(np.conj(self.csr).T)

    @property
    def abs(self):
        return Operator(np.abs(self.csr))

    @property
    def nop(self):
        return self.dag * self

    @staticmethod
    def _get_value(other):
        if isinstance(other, Operator):
            other = other.csr
        return other

    def dot(self, other):
        return Operator(self.csr.dot(self._get_value(other)))

    def __mul__(self, other):
        return Operator(self.csr * self._get_value(other))

    def __rmul__(self, other):
        return Operator(self._get_value(other) * self.csr)

    def __truediv__(self, other):
        return Operator(self.csr / self._get_value(other))

    def __rtruediv__(self, other):
        return Operator(self._get_value(other) / self.csr)

    def __add__(self, other):
        return Operator(self.csr + self._get_value(other))

    def __radd__(self, other):
        return Operator(self._get_value(other) + self.csr)

    def __str__(self):
        return str(self.dense)


class HamiltonOperator:

    def __init__(self, **opkwargs):
        self.operators = opkwargs

    @property
    def keys(self):
        return self.operators.keys()

    def set_operator(self, key, value):
        self.operators.update({key: value})

    def build_operator(self, key, val):
        ops = self.operators[key]
        if hasattr(ops, "__len__"):
            if not hasattr(val, "__len__"):
                val = [val] * len(ops)
            return sum([x * o for x, o in zip(val, ops)])
        else:
            return val * ops

    def build(self, **params):
        ops = list()
        for key in self.keys:
            val = params.get(key, None)
            if val is None:
                val = 0
            ops.append(self.build_operator(key, val))
        return sum(ops)

    def hamiltonian(self, **params):
        return Hamiltonian(self.build(**params).dense)
