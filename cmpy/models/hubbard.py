# -*- coding: utf-8 -*-
"""
Created on 29 Mar 2019
author: Dylan

project: cmpy
version: 1.0

HUBBARD MODEL
"""
import numpy as np
from itertools import product
from cmpy.core import Lattice, State, Hamiltonian


def get_hubbard_states(sites, n=None, spin=None):
    spin_states = list(product([0, 1], repeat=sites))
    states = [State(up, down) for up, down in product(spin_states, repeat=2)]
    for s in states[:]:
        if (n is not None) and (s.num != n):
            states.remove(s)
        elif (spin is not None) and (s.spin != spin):
            states.remove(s)
    return states


def hubbard_hamiltonian(states, eps, t, u, mu=0):
    n = len(states)
    ham = Hamiltonian.zeros(n)
    t_conj = np.conj(t)

    for i in range(n):
        state = states[i]
        ham[i, i] = eps * np.sum(state.num_onsite()) + u * np.sum(state.interaction()) - mu
        for j in range(n):
            if i < j:
                hops = state.check_hopping(states[j])
                if hops is not None:
                    i1, i2 = hops
                    if abs(i1 - i2) == 1:
                        ham[i, j] = t
                        ham[j, i] = t_conj
    return ham


class HubbardModel:

    def __init__(self, shape=(2, 1), eps=0., t=1., u=10, mu=None):
        self.lattice = Lattice.square(shape)
        self.states = list()

        self.eps = eps
        self.t = t
        self.u = u
        self.mu = u / 2 if mu is None else mu
        self.set_filling(self.lattice.n)

    def set_filling(self, n, spin=None):
        self.states = get_hubbard_states(self.lattice.n, n, spin)

    @property
    def state_labels(self):
        return [x.repr for x in self.states]

    def hamiltonian(self):
        return hubbard_hamiltonian(self.states, self.eps, self.t, self.u, self.mu)

    def spectral(self, omegas):
        ham = self.hamiltonian()
        gf = ham.gf(omegas)
        return -np.sum(gf.imag, axis=1)

    def dos(self, omegas):
        return 1/np.pi * self.spectral(omegas)

    def __repr__(self):
        eps = u"\u03b5".encode("utf-8")
        mu = u"\u03bc".encode()
        return f"Hubbard({eps}={self.eps}, t={self.t}, U={self.u}, {mu}={self.mu})"
