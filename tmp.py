# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2018
@author: Dylan Jones

project: tmp$
version: 1.0
"""
import numpy as np
import scipy.linalg as la
from itertools import product
from sciutils import Plot, use_cycler
from cmpy import Hamiltonian
from cmpy.models import HubbardModel, TightBinding
from scipy.sparse import csr_matrix
import colorcet as cc

use_cycler()


def decompose_hamiltonian(ham):
    h, ev = la.eigh(ham)
    ev_inv = ev.conj().T
    return ev_inv, h, ev


def hamiltonian(operators, u, eps_imp, eps, v):
    d_up, d_dw, c_up, c_dw = operators
    ham = eps_imp * (d_up.T * d_up + d_dw.T * d_dw)
    ham += u * (d_up.T * d_up * d_dw.T * d_dw)
    ham += eps * (c_up.T * c_up + c_dw.T * c_dw)
    ham += v * (d_up.T * c_up + d_dw.T * c_dw + c_up.T * d_up + c_dw.T * d_dw)
    return Hamiltonian(ham.todense())


def gf_lehmann(eig_e, eig_states, d_dag, beta, omega, d=None):
    """Outputs the lehmann representation of the greens function
       omega has to be given, as matsubara or real frequencies"""
    ew = np.exp(-beta*eig_e)
    zet = ew.sum()
    G = np.zeros_like(omega)
    basis_create = np.dot(eig_states.T, d_dag.dot(eig_states))
    if d is None:
        tmat = np.square(basis_create)
    else:
        tmat = np.dot(eig_states.T, d.T.dot(eig_states))*basis_create

    tmat = tmat * np.add.outer(ew, ew)
    gap = np.add.outer(-eig_e, eig_e)

    N = eig_e.size
    for i, j in product(range(N), range(N)):
        G += tmat[i, j] / (omega + gap[i, j])
    return G / zet


def nop(cop):
    return cop.T * cop


def destruct_operators(n):
    return [destruct(n, i) for i in range(n)]


def destruct(particles, index):
    """ Creates a fermionic annihilation operator in matrix representation

    The phase +-1 is needed to ensure the operators follow the commutator-rules

    Parameters
    ----------
    particles: int
        Number of particles in the system
    index: int
        index of operator

    Returns
    -------
    mat: csr_matrix
        sparse matrix representation of operator
    """
    n = 2**particles
    mat = np.zeros((n, n))
    flipper = 2**index
    for i in range(n):
        # check if there is a spin at index i
        ispin = (i >> index) & 1
        if ispin == 1:
            phase = 1 if bin(i >> index+1).count('1') % 2 == 0 else -1
            mat[i ^ flipper, i] = phase
    return csr_matrix(mat)


def commutator(o1, o2):
    return o1 * o2 - o2 * o1


def anticommutator(o1, o2):
    return o1 * o2 + o2 * o1


def main():
    eta = 0.01j
    eps, t = 0, 1
    u = 10
    temp = 5
    omegas = np.linspace(-10, 10, 1000)

    states = np.array([[0, 0], [1, 0], [0, 1]])
    operator(states, 0)



    return
    ham = eps*(nop(c1) + nop(c2)) + t*(c1.T*c2 + c2.T*c1)
    ham = Hamiltonian(ham.todense())
    # ham = hamiltonian(operators, 0, eps, eps, t)
    ham.show()

    return
    # eig_e, eig_states = ham.eig()
    # gf_up = gf_lehmann(eig_e, eig_states, operators[0].T, 1/temp, omegas + eta)
    # gf_down = gf_lehmann(eig_e, eig_states, operators[1].T, 1/temp, omegas + eta)

    plot = Plot()
    plot.plot(omegas, -gf_up.imag)
    # plot.plot(omegas, -gf_down.imag)
    plot.show()


if __name__ == "__main__":
    main()
