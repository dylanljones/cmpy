#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24.11.21

@author: Nico Unglert
"""

import numpy as np
from cmpy.models.abc import AbstractSpinModel
import lattpy as lp
import numpy.linalg as la
from cmpy import kron, matshow

si = np.eye(2)
sx = 0.5 * np.array([[0, 1], [1, 0]])
sy = 0.5 * np.array([[0, -1j], [1j, 0]])
sz = 0.5 * np.array([[1, 0], [0, -1]])
sp = np.array([[0, 1], [0, 0]])
sm = np.array([[0, 0], [1, 0]])
pauli = si, sx, sy, sz


class HeisenbergModel(AbstractSpinModel):

    def __init__(self, latt, j=1.):
        super().__init__(latt.num_sites)
        self.latt = latt
        self.j = j
        self.states = self.get_states()
        shape = len(self.states), len(self.states)
        self.hilbert = np.zeros(shape)
        self.ham = np.zeros(shape)
        self.neighbors = [self.latt.neighbors(pos1) for pos1 in range(self.num_sites)]

    def _hamiltonian_data(self):
        for idx1, s1 in enumerate(self.states):
            for pos1 in range(self.num_sites):
                for pos2 in self.neighbors[pos1]:
                    # Diagonal
                    b1 = (s1 >> pos1) & 1  # Bit at index `pos1`
                    b2 = (s1 >> pos2) & 1  # Bit at index `pos2`
                    sign = (-1) ** b1 * (-1) ** b2  # Sign +1 if bits are equal, -1 otherwise
                    yield idx1, idx1, sign * self.j / 4
                    # Off-diagonal
                    # check, if s1 is occupied at pos1 and pos2
                    op = 1 << pos1  # bit-operator at `pos1`
                    occ = s1 & op  # bit value of bit  at `pos1`
                    op2 = 1 << pos2  # bit-operator at `pos2`
                    occ2 = s1 & op2  # bit value of bit  at `pos2`
                    # if occupied at pos1 and pos2, no S+ or S- terms can occur
                    # following lines find the state, to which hopping can occur, considering
                    # the spin situations at pos1 and pos2. Only one possibility!
                    if (occ and not occ2) or (not occ and occ2):
                        # Hopping from `pos1` to `pos2` possible
                        tmp = s1 ^ op  # Annihilate or create state at `pos1`
                        s2 = tmp ^ op2  # create new state with XOR
                        idx2 = self.states.index(s2)  # get index of new state
                        yield idx1, idx2, self.j / 2

    def sz(self, pos):
        for idx1, s1 in enumerate(self.states):
            b1 = (s1 >> pos) & 1  # Bit at index `pos`
            sign = (-1) ** b1  # Sign +1 if spin up, -1 otherwise
            yield idx1, idx1, sign * 1 / 2

    def sp(self, pos):
        for idx1, s1 in enumerate(self.states):
            op = 1 << pos  # bit-operator at `pos1`
            occ = s1 & op  # bit value of bit  at `pos1`
            if occ:
                s2 = s1 ^ op  # Annihilate or create state at `pos1`
                idx2 = self.states.index(s2)  # get index of new state
                yield idx1, idx2, 1

    def sm(self, pos):
        for idx1, s1 in enumerate(self.states):
            op = 1 << pos  # bit-operator at `pos1`
            occ = s1 & op  # bit value of bit  at `pos1`
            if not occ:
                s2 = s1 ^ op  # Annihilate or create state at `pos1`
                idx2 = self.states.index(s2)  # get index of new state
                yield idx1, idx2, 1

    def hamiltonian(self, s=None, states=None, dtype=None):
        for i, j, val in self._hamiltonian_data():
            self.ham[i, j] = val
        return self.ham

    def spins(self, spingenerator):
        for i, j, val in spingenerator:
            self.hilbert[i, j] = val
        return self.hilbert


def get_super_index(i, j, len_j):
    return i * len_j + j


def get_indices(index, len_j):
    return divmod(index, len_j)


class DMRG():

    def __init__(self, size=2) -> None:
        self.size = size

    def one_cycle(self, system, environment):
        size_sys = 2 ** system.num_sites
        size_env = 2 ** environment.num_sites
        ham_sys = system.hamiltonian()
        ham_env = environment.hamiltonian()
        sz_sys = system.spins(system.sz(pos=0))
        sp_sys = system.spins(system.sp(pos=0))
        sm_sys = system.spins(system.sm(pos=0))
        sz_env = environment.spins(system.sz(pos=environment.num_sites - 1))
        sp_env = environment.spins(system.sp(pos=environment.num_sites - 1))
        sm_env = environment.spins(system.sm(pos=environment.num_sites - 1))
        # set up hamiltonian
        ham = (
                kron(ham_sys, np.eye(size_env))
                + kron(np.eye(size_sys), ham_env)
                + kron(sz_sys, sz_env)
                + 0.5 * kron(sp_sys, sm_env)
                + 0.5 * kron(sm_sys, sp_env)
        )
        # matrix.matshow(ham_sys)
        ham_sysvals, ham_sysvecs = la.eigh(ham_sys)

        # determine ground state
        hamvals, hamvecs = la.eigh(ham)
        vector = hamvecs[:, 0]

        # compute density matrix and its eigenvalues, eigenvectors
        psi_mat = vector.reshape((size_sys, size_env))
        dens_mat = np.matmul(psi_mat, psi_mat.T)
        densvals, densvecs = la.eigh(dens_mat)
        # matrix.matshow(dens_mat)

        # calculate and apply rotation and reduction matrix
        cutoff = round(0.7 * size_sys)
        rot_mat = np.flip(densvecs, axis=1)[:, :cutoff]
        print(f"Cutoff error: {1 - np.sum(np.flip(densvals)[:cutoff])}")
        ham_sys = np.matmul(rot_mat.T, np.matmul(ham_sys, rot_mat))
        sz_sys = np.matmul(rot_mat.T, np.matmul(sz_sys, rot_mat))
        sp_sys = np.matmul(rot_mat.T, np.matmul(sp_sys, rot_mat))
        sm_sys = np.matmul(rot_mat.T, np.matmul(sm_sys, rot_mat))
        hamvals, hamvecs = la.eigh(system.ham)
        vector = hamvecs[:, 0]
        print(ham_sysvals, "\n", hamvals)
        return ham_sys, sz_sys, sp_sys, sm_sys

    def init_heis(self, sites=2):
        latt = lp.simple_chain(a=1)
        latt.build(sites-1)
        system = HeisenbergModel(latt)
        environment = HeisenbergModel(latt)
        ham_sys = system.hamiltonian()
        ham_env = environment.hamiltonian()
        sz_sys = system.spins(system.sz(pos=0))
        sp_sys = system.spins(system.sp(pos=0))
        sm_sys = system.spins(system.sm(pos=0))
        sz_env = environment.spins(system.sz(pos=environment.num_sites - 1))
        sp_env = environment.spins(system.sp(pos=environment.num_sites - 1))
        sm_env = environment.spins(system.sm(pos=environment.num_sites - 1))
        return  ham_sys, sz_sys, sp_sys, sm_sys, \
                ham_env, sz_env, sp_env, sm_env

    def dmrg_iter(self, ham_sys, sz_sys, sp_sys, sm_sys,
                  ham_env, sz_env, sp_env, sm_env):
        size_sys = ham_sys.shape[0]
        size_env = ham_env.shape[0]
        ham = (
                kron(ham_sys, np.eye(size_env))
                + kron(np.eye(size_sys), ham_env)
                + kron(sz_sys, sz_env)
                + 0.5 * kron(sp_sys, sm_env)
                + 0.5 * kron(sm_sys, sp_env)
        )
        # matrix.matshow(ham_sys)
        ham_sysvals, ham_sysvecs = la.eigh(ham_sys)

        # determine ground state
        hamvals, hamvecs = la.eigh(ham)
        vector = hamvecs[:, 0]

        # compute density matrix and its eigenvalues, eigenvectors
        psi_mat = vector.reshape((size_sys, size_env))
        dens_mat = np.matmul(psi_mat, psi_mat.T)
        densvals, densvecs = la.eigh(dens_mat)
        # matrix.matshow(dens_mat)

        # calculate and apply rotation and reduction matrix
        cutoff = round(0.7 * size_sys)
        rot_mat = np.flip(densvecs, axis=1)[:, :cutoff]
        print(f"Cutoff error: {1 - np.sum(np.flip(densvals)[:cutoff])}")
        ham_sys = np.matmul(rot_mat.T, np.matmul(ham_sys, rot_mat))
        sz_sys = np.matmul(rot_mat.T, np.matmul(sz_sys, rot_mat))
        sp_sys = np.matmul(rot_mat.T, np.matmul(sp_sys, rot_mat))
        sm_sys = np.matmul(rot_mat.T, np.matmul(sm_sys, rot_mat))
        hamvals, hamvecs = la.eigh(ham_sys)
        print(ham_sysvals, "\n", hamvals)
        return ham_sys, sz_sys, sp_sys, sm_sys

    def dmrg_loop(self):
        matrices = self.init_heis(sites=2)
        i = 0
        while i < 10:
            reduced = self.dmrg_iter(*matrices)
            ham_sysadd = add_site_sys(*reduced)
            ham_envadd = add_site_env(*reduced)
            matrices = ham_sysadd, *create_linker_sys(reduced[0]), \
                        ham_envadd, *create_linker_env(reduced[0])
            print(len(matrices))
            i += 1



def add_site_sys(ham_sys, sz_sys, sp_sys, sm_sys):
    ham_sysadd = (
            kron(ham_sys, si)
            + kron(sz_sys, sz)
            + 0.5 * kron(sp_sys, sm)
            + 0.5 * kron(sm_sys, sp)
    )
    return ham_sysadd

def add_site_env(ham_sys, sz_sys, sp_sys, sm_sys):
    ham_envadd = (
            kron(si, ham_sys)
            + kron(sz, sz_sys)
            + 0.5 * kron(sm, sp_sys)
            + 0.5 * kron(sp, sm_sys)
    )
    return ham_envadd

def create_linker_sys(ham_sys):
    size_sys = ham_sys.shape[0]
    sz_sys = kron(np.eye(size_sys), sz)
    sp_sys = kron(np.eye(size_sys), sp)
    sm_sys = kron(np.eye(size_sys), sm)
    return sz_sys, sp_sys, sm_sys

def create_linker_env(ham_sys):
    size_sys = ham_sys.shape[0]
    sz_sys = kron(sz, np.eye(size_sys))
    sp_sys = kron(sp, np.eye(size_sys))
    sm_sys = kron(sm, np.eye(size_sys))
    return sz_sys, sp_sys, sm_sys

latt = lp.simple_chain(a=1, neighbors=1)
latt.build(2)
heis = HeisenbergModel(latt)
ssz = heis.spins(heis.sz(pos=0))
print(sz)
dmrg = DMRG()
# ham_sys, sz_sys, sp_sys, sm_sys = dmrg.one_cycle(heis, heis)
# sys_new = add_site_sys(*dmrg.one_cycle(heis, heis))
# matshow(sys_new)
dmrg.dmrg_loop()