# -*- coding: utf-8 -*-
"""
Created on 12 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0

SINGLE-IMPURITY-ANDERSON-MODEL (SIAM)
"""
import numpy as np
from itertools import product
from cmpy.core import State, Hamiltonian, gf_lehmann


def get_siam_states(n_bath=1, n=None, spin=None):
    spin_states = list(product([0, 1], repeat=n_bath + 1))
    states = [State(up, down) for up, down in product(spin_states, repeat=2)]
    for s in states[:]:
        # if any(s.double[1:]):
        #    states.remove(s)
        if (n is not None) and (s.num != n):
            states.remove(s)
        elif (spin is not None) and (s.spin != spin):
            states.remove(s)
    return states


def siam_hamiltonian(states, eps_d, u, eps, v, mu=0.):
    n = len(states)
    energies = np.append([eps_d], eps)
    v = np.asarray(v)
    v_conj = np.conj(v)

    ham = Hamiltonian.zeros(n, dtype="complex")
    for i in range(n):
        state = states[i]

        num_eps = state.num_onsite()
        ham[i, i] = np.dot(num_eps, energies) - mu
        if state.interaction()[0]:
            ham[i, i] += u
        # ham[i, i] += u * np.sum(state.interaction())
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


def siam_hamiltonian_free(eps_d, u, eps, v, mu=0.):
    energies = np.append([eps_d], eps)
    n = len(energies)
    v = np.asarray(v)
    v_conj = np.conj(v)
    ham = Hamiltonian.zeros(n, dtype="complex")
    for i in range(n):
        ham[i, i] = energies[i]
        if i > 0:
            ham[0, i] = v[i-1]
            ham[i, 0] = v_conj[i-1]
    return ham


class Siam:

    def __init__(self, eps_imp=0., u=10., eps=0., v=1., mu=None):
        self.u = u
        self.eps_imp = eps_imp
        self.eps = np.asarray(eps) if hasattr(eps, "__len__") else np.array([eps])
        self.v = np.asarray(v) if hasattr(v, "__len__") else np.array([v])
        self.mu = u / 2 if mu is None else mu

        self.states = list()
        self.n_bath = len(self.eps)
        self.set_filling(self.n_bath + 1)


    @property
    def state_labels(self):
        return [x.label() for x in self.states]

    @property
    def n_sites(self):
        return self.n_bath + 1

    def update_bath_hopping(self, v_new):
        v_new = np.asarray(v_new if hasattr(v_new, "__len__") else np.array([v_new]))
        if self.n_bath != len(v_new):
            raise ValueError("Bath-site dimensions don't match")
        self.v = v_new

    def update_bath_energy(self, eps_new):
        eps_new = np.asarray(eps_new if hasattr(eps_new, "__len__") else np.array([eps_new]))
        if self.n_bath != len(eps_new):
            raise ValueError("Bath-site dimensions don't match")
        self.eps = eps_new

    def update_bath_sites(self, eps, v):
        self.update_bath_energy(eps)
        self.update_bath_hopping(v)

    def set_filling(self, n, spin=None):
        self.states = get_siam_states(self.n_bath, n, spin)

    def sort_states(self, indices):
        if len(indices) != len(self.states):
            raise ValueError(f"Number of indices doesn't match number of states: {len(indices)}!={len(self.states)}")
        self.states = [self.states[i] for i in indices]


    # ==============================================================================================

    def hybridization(self, omega):
        return self.v**2 / (omega + self.mu - self.eps)

    def hamiltonian(self):
        return siam_hamiltonian(self.states, self.eps_imp, self.u, self.eps, self.v, 0)

    def hamiltonian_free(self):
        return siam_hamiltonian_free(self.eps_imp, self.u, self.eps, self.v, 0)

    def gf_imp_free(self, omegas):
        return gf_lehmann(self.hamiltonian_free(), omegas, mu=self.mu)

    def gf_imp(self, omegas):
        return gf_lehmann(self.hamiltonian(), omegas, mu=self.mu)

    def self_energy(self, omega):
        ham = self.hamiltonian()
        gf = np.sum(ham.gf(omega))
        delta = self.hybridization(omega)
        return 1/(omega + self.mu - self.eps_imp - delta - gf)

    def spectral(self, omegas):
        ham = self.hamiltonian()
        gf = ham.gf(omegas)
        return -np.sum(gf.imag, axis=1)

    def dos(self, omegas):
        return 1/np.pi * self.spectral(omegas)

    # ==============================================================================================

    def show_hamiltonian(self):
        ham = self.hamiltonian()
        ham.show(basis_labels=self.state_labels)

    def __str__(self):
        eps_str = ", ".join([f"{x:.1f}" for x in self.eps])
        v_str = ", ".join([f"{x:.1f}" for x in self.v])
        string = f"SIAM: eps_imp={self.eps_imp:.1f}, u={self.u:.1f}, eps={eps_str}, v={v_str}"
        return string
