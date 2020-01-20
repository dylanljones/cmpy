# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2019
author: Dylan Jones

project: dmft
version: 1.0
"""
import numpy as np
from itertools import product
from scipy.sparse import csr_matrix

EMPTY = "."
UP_CHAR = "↑"
DN_CHAR = "↓"
UD_CHAR = "⇅"  # "d"


def binstr(st, fill=0):
    return f"{st:0{fill}b}"


def one_particle_state(i):
    return 1 << i


def particle_op(st, i):
    return (1 << i) & st


def particle_number(st):
    return bin(st).count("1")


def create(st, i):
    op = 1 << i
    if op & st == 0:
        return op ^ st


def annihilate(st, i):
    op = 1 << i
    if op & st == 1:
        return op ^ st


def spin_basis(num_sites, n=None):
    max_int = int("1" * num_sites, 2)
    if n is None:
        return list(range(max_int + 1))
    else:
        states = list()
        for st in range(max_int + 1):
            if particle_number(st) == n:
                states.append(st)
        return states


def iter_basis_states(n_sites):
    return product(range(2 ** n_sites), repeat=2)


def state_label(up, dn, n_sites):
    chars = list()
    for i in range(n_sites):
        char = EMPTY
        u = up >> i & 1
        d = dn >> i & 1
        if u and d:
            char = UD_CHAR
        elif u:
            char = UP_CHAR
        elif d:
            char = DN_CHAR
        chars.append(char)
    return "".join(chars)[::-1]


def check_filling(up, dn, particles=None):
    if particles is None:
        return True
    if not hasattr(particles, "__len__"):
        particles = [particles]
    n = bin(up).count("1") + bin(dn).count("1")
    return n in particles


def check_spin(up, dn, spin_values=None):
    if spin_values is None:
        return True
    if not hasattr(spin_values, "__len__"):
        spin_values = [spin_values]
    s = 0.5 * (bin(up).count("1") - bin(dn).count("1"))
    return s in spin_values


def check_sector(up, down, n=None, s=None):
    return check_filling(up, down, n) and check_spin(up, down, s)

# =========================================================================


class State:

    BITS = 2
    UP, DN = 0, 1
    BITSTR_ORDER = +1

    __slots__ = ["up", "dn"]

    def __init__(self, up, dn):
        self.up = up if not isinstance(up, str) else int(up, 2)
        self.dn = dn if not isinstance(dn, str) else int(dn, 2)

    @staticmethod
    def set_bitcount(n):
        State.BITS = n

    def __format__(self, format_spec):
        up_str = f"{self.up:{format_spec}}"
        dn_str = f"{self.dn:{format_spec}}"
        return f"State({up_str[::-self.BITSTR_ORDER]}, {dn_str[::self.BITSTR_ORDER]})"

    def __repr__(self):
        up_str = f"{binstr(self.up, self.BITS)}"
        dn_str = f"{binstr(self.dn, self.BITS)}"
        return f"State({up_str[::self.BITSTR_ORDER]}, {dn_str[::self.BITSTR_ORDER]})"

    def label(self, n_sites=None):
        n_sites = self.BITS if n_sites is None else n_sites
        return state_label(self.up, self.dn, n_sites)[::self.BITSTR_ORDER]

    def __eq__(self, other):
        return self.up == other.up and self.dn == other.dn

    @property
    def n_up(self):
        return bin(self.up).count("1")

    @property
    def n_dn(self):
        return bin(self.dn).count("1")

    @property
    def n(self):
        return bin(self.up).count("1") + bin(self.dn).count("1")

    @property
    def s_up(self):
        return +0.5 * bin(self.up).count("1")

    @property
    def s_dn(self):
        return -0.5 * bin(self.dn).count("1")

    @property
    def s(self):
        return 0.5 * (bin(self.up).count("1") - bin(self.dn).count("1"))

    def get_spinstate(self, sigma):
        return self.up if sigma == self.UP else self.dn

    def get_particles(self, sigma):
        return self.n_up if sigma == self.UP else self.n_dn

    def get_spin(self, sigma):
        return self.s_up if sigma == self.UP else self.s_dn

    def get_indices(self, sigma):
        return [i for i, char in enumerate(binstr(self.get_spinstate(sigma))) if char == "1"]

    def create(self, i, sigma):
        op = 1 << i
        if sigma == self.UP:
            if op & self.up == 0:
                return State(op ^ self.up, self.dn)
        else:
            if op & self.dn == 0:
                return State(self.up, op ^ self.dn)

    def annihilate(self, i, sigma):
        op = 1 << i
        if sigma == self.UP:
            if op & self.up == 1:
                return State(op ^ self.up, self.dn)
        else:
            if op & self.dn == 1:
                return State(self.up, op ^ self.dn)

    def spin_hopping_indices(self, other, sigma):
        st1 = self.get_spinstate(sigma)
        st2 = other.get_spinstate(sigma)
        diff = st1 ^ st2
        if diff:
            if bin(st1).count("1") == bin(st2).count("1"):
                indices = [i for i, char in enumerate(binstr(diff, 0)) if char == "1"]
                if len(indices) == 2:
                    return indices

    def hopping_indices(self, other):
        if self.s != other.s:
            return None
        up_indices = self.spin_hopping_indices(other, self.UP)
        dn_indices = self.spin_hopping_indices(other, self.DN)
        if up_indices is not None and dn_indices is None:
            return up_indices
        elif dn_indices is not None and up_indices is None:
            return dn_indices
        else:
            return None


class Basis:

    def __init__(self, n_sites, n=None, s=None, states=None):
        State.set_bitcount(n_sites)
        self.n_sites = n_sites
        if states is None:
            states = self.get_particle_states(n, s)
        self.states = states
        # self.sort()

    def get_particle_states(self, n=None, s=None):
        states = list()
        if n != 1:
            for up, dn in iter_basis_states(self.n_sites):
                if check_sector(up, dn, n, s):
                    states.append(State(up, dn))
        else:
            s = s or +0.5
            for i in range(self.n_sites):
                spin_state = 1 << i
                state = State(spin_state, 0) if s == 0.5 else State(0, spin_state)
                states.append(state)
        return states

    def subbasis(self, n=None, s=None):
        return Basis(self.n_sites, n, s)

    @property
    def n(self):
        return len(self.states)

    @property
    def labels(self):
        return [s.label(self.n_sites) for s in self.states]

    def __str__(self):
        string = "Basis:"
        for label in self.labels:
            string += f"\n  {label}"
        return string

    def __getitem__(self, item):
        if hasattr(item, "__len__"):
            return [self.states[i] for i in item]
        else:
            return self.states[item]

    def sort(self, key=None):
        if key is None:
            key = lambda s: s.n
        self.states.sort(key=key)

    def get(self, n=None, s=None):
        states = list()
        for state in self.states:
            if state.check_sector(n, s):
                states.append(state)
        return states

    def index(self, state):
        return self.states.index(state)

    def c_operators(self):
        return annihilation_operators(self)


def _ordering_phase(spin_state, i):
    particles = bin(spin_state >> i + 1)[2:].count("1")
    return 1 if particles % 2 == 0 else -1


def annihilation_op_csr(basis, idx, spin):
    n = basis.n
    row, col, data = list(), list(), list()
    for j, state in enumerate(basis):
        other = state.annihilate(idx, spin)
        if other is not None:
            try:
                i = basis.index(other)
                spin_state = state.spin(spin)
                val = _ordering_phase(spin_state, idx)
                row.append(i), col.append(j), data.append(val)
            except ValueError:
                pass
    return csr_matrix((data, (row, col)), shape=(n, n), dtype="int")


class Operator:

    def __init__(self, array=None):
        if isinstance(array, Operator):
            array = array.csr
        self.csr = csr_matrix(array)

    @classmethod
    def c(cls, basis, idx, spin):
        mat = annihilation_op_csr(basis, idx, spin)
        return cls(mat)

    @classmethod
    def cdag(cls, idx, states, spin):
        return cls.c(idx, states, spin).dag

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

    def dense(self, **params):
        return self.build(**params).todense()


def annihilation_operators(basis):
    up_ops = [Operator.c(basis, i, State.UP) for i in range(basis.n_sites)]
    dn_ops = [Operator.c(basis, i, State.DN) for i in range(basis.n_sites)]
    return up_ops, dn_ops
