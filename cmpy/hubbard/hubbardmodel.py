# -*- coding: utf-8 -*-
"""
Created on 29 Mar 2019
author: Dylan

project: cmpy2
version: 1.0
"""
import numpy as np
from itertools import product
from ..core import Lattice, Hamiltonian
from .state import State


# =========================================================================
# HUBBARD MODEL
# =========================================================================


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
        ham[i, i] = eps * state.n_onsite() + u * state.n_interaction() - mu
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

    def get_siam(self, eps=0, v=1):
        siam = Siam(eps_d=self.eps, u=self.u, eps=eps, v=v, mu=self.mu)
        return siam

    def __repr__(self):
        eps = u"\u03b5".encode("utf-8")
        mu = u"\u03bc".encode()
        return f"Hubbard({eps}={self.eps}, t={self.t}, U={self.u}, {mu}={self.mu})"


# =========================================================================
# SINGLE-IMPURITY-ANDERSON-MODEL (SIAM)
# =========================================================================


def get_siam_states(n_bath=1, n=None, spin=None):
    spin_states = list(product([0, 1], repeat=n_bath + 1))
    states = [State(up, down) for up, down in product(spin_states, repeat=2)]
    for s in states[:]:
        if any(s.double[1:]):
            states.remove(s)
        elif (n is not None) and (s.num != n):
            states.remove(s)
        elif (spin is not None) and (s.spin != spin):
            states.remove(s)
    return states


def siam_hamiltonian(states, eps_d, u, eps, v, mu=0):
    n = len(states)
    energies = np.append([eps_d], eps)
    v = np.asarray(v)
    v_conj = np.conj(v)

    ham = Hamiltonian.zeros(n, dtype="complex")
    for i in range(n):
        state = states[i]
        idx = np.where(state.single)[0]
        ham[i, i] = np.sum(energies[idx]) + u * state.n_interaction() - mu
        for j in range(n):
            if i < j:
                hops = state.check_hopping(states[j])
                if hops is not None:
                    i1, i2 = sorted(hops)
                    if i1 == 0:
                        idx = i2 - 1
                        ham[i, j] = v[idx]
                        ham[j, i] = v_conj[idx]
    return ham


class Siam:

    def __init__(self, eps_d=0., u=10., eps=0., v=1., mu=None):
        self.u = u
        self.mu = u / 2 if mu is None else mu
        self.eps_d = eps_d
        self.eps = np.asarray(eps) if hasattr(eps, "__len__") else np.array([eps])
        self.v = np.asarray(v) if hasattr(v, "__len__") else np.array([v])
        self.n_bath = len(self.eps)

        self.states = list()
        self.set_filling(self.n_bath + 1)

    def update_bath_sites(self, eps, v):
        if self.n_bath != len(self.eps):
            raise ValueError("Bath-site dimensions don't match")
        self.eps = np.asarray(eps) if hasattr(eps, "__len__") else np.array([eps])
        self.v = np.asarray(v) if hasattr(v, "__len__") else np.array([v])

    def set_filling(self, n, spin=None):
        self.states = get_siam_states(self.n_bath, n, spin)

    @property
    def state_labels(self):
        return [x.repr for x in self.states]

    def hybridization(self, omega):
        return self.v**2 / (omega + self.mu - self.eps)

    def hamiltonian(self):
        return siam_hamiltonian(self.states, self.eps_d, self.u, self.eps, self.v, self.mu)

    def self_energy(self, omega):
        ham = self.hamiltonian()
        gf = np.sum(ham.gf(omega))
        delta = self.hybridization(omega)
        return 1/(omega + self.mu - self.eps_d - delta - gf)

    def spectral(self, omegas):
        ham = self.hamiltonian()
        gf = ham.gf(omegas)
        return -np.sum(gf.imag, axis=1)

    def dos(self, omegas):
        return 1/np.pi * self.spectral(omegas)

    def __str__(self):
        eps_str = ", ".join([f"{x:.1f}" for x in self.eps])
        v_str = ", ".join([f"{x:.1f}" for x in self.v])
        string = f"SIAM: eps_d={self.eps_d:.1f}, u={self.u:.1f}, eps={eps_str}, v={v_str}"
        return string
