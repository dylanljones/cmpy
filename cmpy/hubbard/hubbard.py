# -*- coding: utf-8 -*-
"""
Created on 29 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
from cmpy.core import Hamiltonian, square_lattice
from .basis import Basis


class HubbardModel:

    def __init__(self, eps=0., t=1., u=None, mu=None):
        self.lattice = square_lattice()
        self.basis = None
        self.states = list()

        self.eps = eps
        self.t = t
        self.u = 10 * t if u is None else u
        self.mu = self.u / 2 if mu is None else mu

        self.build(x=2)

    def build(self, x=2, y=1):
        self.lattice.build((x, y))
        self.basis = Basis(int(x * y))

    def hamiltonian(self, n=None, spin=None):
        states = self.basis.get_states(n, spin)
        ham = Hamiltonian.zeros(len(states))
        for i, j in ham.iter_indices():
            state = states[i]
            if i == j:
                ham.set_energy(i, self.eps)
            if j < i:
                hops = state.check_hopping(states[j])
                if hops is not None:
                    i1, i2 = hops
                    if abs(i1 - i2) == 1:
                        ham.set_hopping(i, j, self.t)
        for i in range(ham.n):
            state = states[i]
            ham.set_energy(i, self.u * state.n_interaction())
        self.states = states
        return ham

    def __repr__(self):
        return f"Hubbard(\u03b5={self.eps}, t={self.t}, U={self.u}, \u03bc={self.mu})"
