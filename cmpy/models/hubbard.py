# -*- coding: utf-8 -*-
"""
Created on 29 Mar 2019
author: Dylan

project: cmpy
version: 1.0

HUBBARD MODEL
"""
import numpy as np
from cmpy.core import Lattice, fock_basis, Hamiltonian


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


def hubbard_hamiltonian2(n_sites, states, u, eps, t):
    n = len(states)
    ham = Hamiltonian.zeros(n)
    for i in range(n):
        s1 = states[i]
        occ = s1.occupations(n_sites)
        # On-site energy
        idx = np.where(occ > 0)[0]
        ham[i, i] = np.sum(eps * occ[idx])
        # Impurity interaction
        ham[i, i] += u * len(occ[occ == 2])
        # Hopping
        for j in range(i+1, n):
            s2 = states[j]
            idx = s1.check_hopping(s2)
            if idx is not None:
                if abs(idx[0] - idx[1]) == 2:
                    ham[i, j] = t
                    ham[j, i] = t
    return ham


class HubbardModel:

    def __init__(self, shape=(2, 1), u=10., eps=0., t=1., mu=None):
        self.lattice = Lattice.square(shape)
        self.states = fock_basis(self.lattice.n)

        self.u = u
        self.eps = eps
        self.t = t
        self.mu = u / 2 if mu is None else mu
        # self.set_filling(self.lattice.n)

    @property
    def n(self):
        return self.lattice.n

    @property
    def state_labels(self):
        return [x.repr for x in self.states]

    def hamiltonian(self):
        return hubbard_hamiltonian2(self.n, self.states, self.u, self.eps, self.t)

    def spectral(self, omegas):
        ham = self.hamiltonian()
        gf = ham.gf(omegas)
        return -np.sum(gf.imag, axis=1)

    def dos(self, omegas):
        return 1/np.pi * self.spectral(omegas)

    def __repr__(self):
        eps = u"\u03b5".encode("utf-8")
        mu = u"\u03bc".encode("utf-8")
        return f"Hubbard(U={self.u}, {eps}={self.eps}, t={self.t}, {mu}={self.mu})"
