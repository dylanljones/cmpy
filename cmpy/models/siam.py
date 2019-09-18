# -*- coding: utf-8 -*-
"""
Created on 15 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from cmpy.core import fock_basis, annihilators, free_basis, state_indices
from cmpy.core.utils import ensure_array
from cmpy.core.greens import greens_function_free, greens_function, self_energy
from cmpy import Hamiltonian, HamiltonOperator
# from cmpy.dmft.two_site import self_energy, gf_lehmann


def siam_hamiltonian(states, u, eps_imp, eps_bath, v):
    n_sites = 2
    energies = np.array([eps_imp, eps_bath])
    n = len(states)
    ham = Hamiltonian.zeros(n)
    for i in range(n):
        s1 = states[i]

        # On-site energy
        occ = s1.occupations(n_sites)
        idx = np.where(occ > 0)[0]
        ham[i, i] = np.sum(energies[idx] * occ[idx])

        # Impurity interaction
        if occ[0] == 2:
            ham[i, i] += u

        for j in range(i+1, n):
            s2 = states[j]
            idx = s1.check_hopping(s2)
            if idx is not None:
                ham[i, j] = v
                ham[j, i] = v
    return ham


# =========================================================================


def siam_operator(operators):
    c0u, c0d = operators[:2]
    bath_ops = operators[2:]
    u_op = c0u.T * c0u * c0d.T * c0d
    eps_imp_op = c0u.T * c0u + c0d.T * c0d
    eps_list, v_list = list(), list()
    for i in range(0, len(bath_ops), 2):
        ciu, cid = bath_ops[i:i+2]
        eps = ciu.T * ciu + cid.T * cid
        hop = (c0u.T * ciu + ciu.T * c0u) + (c0d.T * cid + cid.T * c0d)
        eps_list.append(eps)
        v_list.append(hop)
    return HamiltonOperator(u=u_op, eps_imp=eps_imp_op, eps_bath=eps_list, v=v_list)


def free_siam_operator(operators):
    c0 = operators[0]
    bath_ops = operators[1:]
    eps_imp_op = c0.T * c0
    eps_list, v_list = list(), list()
    for i in range(0, len(bath_ops), 1):
        ci = bath_ops[i]
        eps = ci.T * ci
        hop = c0.T * ci + ci.T * c0
        eps_list.append(eps)
        v_list.append(hop.abs)
    return HamiltonOperator(eps_imp=eps_imp_op, eps_bath=eps_list, v=v_list)


# =========================================================================

class Siam:

    UP, DOWN = 0, 1

    def __init__(self, u, eps_imp, eps_bath, v, mu=None, beta=0.):
        """ Initilizes the single impurity Anderson model

        Parameters
        ----------
        u: float
        eps_imp: float
        eps_bath: float or array_like
        v: float or array_like
        mu: float, optional
        """
        self.u = u
        self.eps_imp = eps_imp
        self.eps_bath = np.asarray(ensure_array(eps_bath))
        self.v = np.asarray(ensure_array(v))
        self.mu = u / 2 if mu is None else mu
        self.beta = beta

        self.n = len(self.eps_bath) + 1

        self.states = None
        self.ops = None
        self.ham_op = None

        self.free_states = free_basis(self.n)
        self.free_ops = annihilators(self.free_states, self.n)
        self.free_ham_op = free_siam_operator(self.free_ops)

        self.set_basis()

    @property
    def n_bath(self):
        return self.n - 1

    @property
    def c_up(self):
        return self.ops[::2]

    @property
    def c_down(self):
        return self.ops[1::2]

    def _build_operators(self):
        self.ops = annihilators(self.states, 2 * self.n)
        self.ham_op = siam_operator(self.ops)

    def set_basis(self, particles=None):
        allstates = fock_basis(self.n)
        n = len(allstates)
        states = allstates.copy()
        if particles is not None:
            for i in range(n):
                if allstates[i].particles not in particles:
                    states.remove(allstates[i])
        self.states = states
        self._build_operators()

    def sort_states(self, *args, **kwargs):
        self.states = sorted(self.states, *args, **kwargs)
        self._build_operators()

    def update_bath_energy(self, eps_bath):
        eps_bath = ensure_array(eps_bath)
        if len(eps_bath) != self.n_bath:
            raise ValueError("Number of bath-parameters doesn't match existing ones")
        self.eps_bath = np.asarray(ensure_array(eps_bath))

    def update_hybridization(self, v):
        v = ensure_array(v)
        if len(v) != self.n_bath:
            raise ValueError("Number of bath-parameters doesn't match existing ones")
        self.v = np.asarray(ensure_array(v))

    def update_bath(self, eps_bath, v):
        self.update_bath_energy(eps_bath)
        self.update_hybridization(v)

    def hybridization(self, z):
        delta = self.v[np.newaxis, :]**2 / (z + self.mu - self.eps_bath[np.newaxis, :])
        return delta.sum(axis=0)

    def hamiltonian(self):
        ham = self.ham_op.build(u=self.u, eps_imp=self.eps_imp, eps_bath=self.eps_bath, v=self.v)
        return Hamiltonian(ham.dense)

    def hamiltonian_manual(self):
        return siam_hamiltonian(self.states, self.u, self.eps_imp, self.eps_bath[0], self.v[0])

    def hamiltonian_free(self, zerostate=False):
        ham = self.free_ham_op.build(eps_imp=self.eps_imp, eps_bath=self.eps_bath, v=self.v)
        ham = ham.dense[1:, 1:] if not zerostate else ham.dense
        return Hamiltonian(ham)

    def impurity_gf(self, z, spin=0):
        ham = self.hamiltonian()
        return greens_function(ham, self.ops[spin], z + self.mu, self.beta)

    def impurity_gf_free2(self, z):
        ham0 = self.hamiltonian_free()
        return greens_function_free(ham0, z + self.mu)[0]

    def impurity_gf_free(self, z):
        return 1/(z + self.mu - self.eps_imp - self.hybridization(z))

    def state_labels(self):
        return [s.label(self.n) for s in self.states]

    def show_hamiltonian(self, show=True):
        ham = self.hamiltonian()
        ham.show(show, show_values=True, labels=self.state_labels())
