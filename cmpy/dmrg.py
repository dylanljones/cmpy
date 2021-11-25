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
import matplotlib.pyplot as plt
import time

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
            self.ham[i, j] += val
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

    def __init__(self, sites=2, max_iter=500, e_conv=1e-5, cutoff=1e-6) -> None:
        self.gs_energies = np.zeros(max_iter)
        self.max_iter = max_iter
        self.sites = sites
        self.e_conv = e_conv
        self.iter = 0
        self.cutoff = cutoff

    def init_heis(self, sites=2):
        latt = lp.simple_chain(a=1)
        latt.build(sites - 1)
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
        return ham_sys, sz_sys, sp_sys, sm_sys, \
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
        self.gs_energies[self.iter] = hamvals[0] / self.sites

        # compute density matrix and its eigenvalues, eigenvectors
        psi_mat = vector.reshape((size_sys, size_env))
        dens_mat = np.matmul(psi_mat, psi_mat.T)
        densvals, densvecs = la.eigh(dens_mat)

        # calculate and apply rotation and reduction matrix
        # cutoff = round(0.7 * size_sys)
        cutoff = np.sum(densvals > self.cutoff)
        rot_mat = np.flip(densvecs, axis=1)[:, :cutoff]
        print(f"Cutoff error: {1 - np.sum(densvals[::-1][:cutoff])}")
        ham_sys = np.matmul(rot_mat.T, np.matmul(ham_sys, rot_mat))
        sz_sys = np.matmul(rot_mat.T, np.matmul(sz_sys, rot_mat))
        sp_sys = np.matmul(rot_mat.T, np.matmul(sp_sys, rot_mat))
        sm_sys = np.matmul(rot_mat.T, np.matmul(sm_sys, rot_mat))
        hamvals, hamvecs = la.eigh(ham_sys)
        # print(ham_sysvals[:5], "\n", hamvals[:5])
        return ham_sys, sz_sys, sp_sys, sm_sys

    def dmrg_loop(self):
        matrices = self.init_heis(sites=self.sites / 2)
        ediff = np.inf
        e_previous = 100
        while ediff > self.e_conv and (self.iter < self.max_iter):
            self.sites += 2
            reduced = self.dmrg_iter(*matrices)
            ham_sysadd = add_site_sys(*reduced)
            ham_envadd = add_site_env(*reduced)
            matrices = ham_sysadd, *create_linker_sys(reduced[0]), \
                       ham_envadd, *create_linker_env(reduced[0])
            print("--------------- end of iteration ----------------")
            ediff = abs(e_previous - self.gs_energies[self.iter])
            e_previous = self.gs_energies[self.iter]
            self.iter += 1
        fig, ax = plt.subplots()
        ax.plot(self.gs_energies[1:self.iter])
        plt.show()


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

def XXZ_Ham(J, Jz, L):
    Sigx = 0.5 * np.array([[0., 1.], [1., 0.]])
    Sigy = 0.5j * np.array([[0., -1.], [1., 0.]])
    Sigz = 0.5 * np.array([[1., 0.], [0., -1.]])
    Sigp = Sigx + 1j * Sigy
    Sigm = Sigx - 1j * Sigy

    SigXX = kron(Sigx, Sigx)
    SigYY = kron(Sigy, Sigy)
    SigZZ = kron(Sigz, Sigz)
    SigPMMP = 0.5 * kron(Sigp, Sigm) + 0.5 * kron(Sigm, Sigp)
    H_Heis = np.array(np.zeros((2 ** L, 2 ** L), dtype='complex'))
    for i1 in np.arange(L - 1):
        Hij = (SigPMMP + Jz * SigZZ)
        for i2 in np.arange(i1):
            Hij = kron(np.eye(2), Hij)
        for i2 in np.arange(i1 + 2, L):
            Hij = kron(Hij, np.eye(2))
        H_Heis = H_Heis + Hij
    return H_Heis


class firat_dmrg:

    def __init__(self, sites=2, max_iter=500, e_conv=1e-5, cutoff=1e-6, J=1, Jz=1) -> None:
        self.gs_energies = np.zeros(max_iter)
        self.max_iter = max_iter
        self.sites = sites
        self.e_conv = e_conv
        self.iter = 0
        self.cutoff = cutoff

    J = 1.
    Jz = 1.
    L = sites
    HA = XXZ_Ham(J, Jz, L)
    HB = HA

    Sigx = 0.5 * np.array([[0., 1.], [1., 0.]])
    Sigy = 0.5j * np.array([[0., -1.], [1., 0.]])
    Sigz = 0.5 * np.array([[1., 0.], [0., -1.]])

    SxL, SyL, SzL, SxR, SyR, SzR = Sigx, Sigy, Sigz, Sigx, Sigy, Sigz
    error = 100.
    MaxIteration = 1000
    eint = 100
    eA = 100
    energies = np.zeros((MaxIteration - L,), dtype='double')
    i0 = 0

    def Enlarge2Site(self, HA, HB, SxL, SyL, SzL, SxR, SyR, SzR, Jz, cutoff, i0):
        Sigx = 0.5 * np.array([[0., 1.], [1., 0.]])
        Sigy = 0.5j * np.array([[0., -1.], [1., 0.]])
        Sigz = 0.5 * np.array([[1., 0.], [0., -1.]])

        ## (1)
        HAL = kron(SxL, Sigx) + kron(SyL, Sigy) + Jz * kron(SzL, Sigz)
        HRB = kron(Sigx, SxL) + kron(Sigy, SyL) + Jz * kron(Sigz, SzL)

        ## (2)
        HLR_local = kron(Sigx, Sigx) + kron(Sigy, Sigy) + Jz * kron(Sigz, Sigz)
        HLR = kron(kron(np.eye(int(HA.shape[0])), HLR_local), np.eye(int(HB.shape[0])))

        ## (3)
        dimSxL = np.prod(SxL.shape[0])
        HApAp = kron(HA, np.eye(2)) + kron(np.eye(int(HA.shape[0] / dimSxL)), HAL)
        HBpBp = kron(np.eye(2), HA) + kron(HRB, np.eye(int(HB.shape[0] / dimSxL)))

        ## (4)
        SxL = kron(np.eye(int(HA.shape[0])), Sigx)
        SyL = kron(np.eye(int(HA.shape[0])), Sigy)
        SzL = kron(np.eye(int(HA.shape[0])), Sigz)
        SxR = kron(Sigx, np.eye(int(HB.shape[0])))
        SyR = kron(Sigy, np.eye(int(HB.shape[0])))
        SzR = kron(Sigz, np.eye(int(HB.shape[0])))

        ## (5)
        H = kron(HApAp, np.eye(int(HBpBp.shape[0]))) + kron(np.eye(int(HApAp.shape[0])), HBpBp) + HLR

        ei, vi = la.eigh(H)
        ndee1 = np.argsort(ei)
        ei = ei[ndee1]
        vi = vi[:, ndee1]

        v0 = np.reshape(vi[:, 0], (np.prod(vi[:, 0].shape), 1))
        psi0 = np.reshape(v0, (int(np.sqrt(H.shape[0])), int(np.sqrt(H.shape[0]))))

        rhoA = np.dot(psi0, np.conj(psi0.T))

        ## (6)
        eA, vA = la.eigh(rhoA)
        ndeeA = np.argsort(-eA)
        eA = eA[ndeeA]
        vA = vA[:, ndeeA]

        indMax = sum(eA > cutoff)
        UTr = vA[:, np.arange(indMax)]
        eATr = eA[np.arange(indMax)]

        ## (7)
        HA = np.dot(np.conj(UTr.T), np.dot(HApAp, UTr))
        HB = np.dot(np.conj(UTr.T), np.dot(HBpBp, UTr))

        SxL = np.dot(np.conj(UTr.T), np.dot(SxL, UTr))
        SyL = np.dot(np.conj(UTr.T), np.dot(SyL, UTr))
        SzL = np.dot(np.conj(UTr.T), np.dot(SzL, UTr))

        SxR = np.dot(np.conj(UTr.T), np.dot(SxR, UTr))
        SyR = np.dot(np.conj(UTr.T), np.dot(SyR, UTr))
        SzR = np.dot(np.conj(UTr.T), np.dot(SzR, UTr))

        return HA, HB, SxL, SyL, SzL, SxR, SyR, SzR, ei[0] / (2 * i0 + 6.), eA[0]

    def firat_loop(self):
        while (error > e_conv) and (i0 < MaxIteration - self.sites):
            HA, HB, SxL, SyL, SzL, SxR, SyR, SzR, energies[i0], eA = Enlarge2Site(HA, HB, SxL, SyL, SzL, SxR, SyR, SzR, Jz,
                                                                                  cutoff, i0)
            error = abs((eint - energies[i0]))
            eint = energies[i0]
            i0 = i0 + 1
            print(f"\r {i0}", end="")
            # (H.shape, HAL.shape,HBR.shape,SxR.shape,SxL.shape,vATr.shape,indMax)


def main():
    dmrg = DMRG(sites=2, e_conv=1e-5, cutoff=3e-5)
    start = time.process_time()
    dmrg.dmrg_loop()
    print(time.process_time() - start)

    start = time.process_time()
    firat(sites=2, e_conv=1e-5, cutoff=3e-5)
    print(time.process_time() - start)


if __name__ == "__main__":
    main()
