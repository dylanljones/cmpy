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
from sciutils import Plot
from cmpy import Hamiltonian
from cmpy.models import HubbardModel, TightBinding
from scipy.sparse import csr_matrix
import colorcet as cc

def decompose_hamiltonian(ham):
    h, ev = la.eigh(ham)
    ev_inv = ev.conj().T
    return ev_inv, h, ev


def gf_lehmann(ham, omega, mu=0., temp=1):
    """ Calculate the greens function of the given Hamiltonian

    Parameters
    ----------
    ham: array_like
        Hamiltonian matrix
    omega: complex or array_like
        Energy e+eta of the greens function (must be complex!)
    mu: float, default: 0
        Chemical potential of the system

    Returns
    -------
    greens: np.ndarray
    """
    omega = np.asarray(omega)
    beta = 1 / temp
    # Calculate eigenvalues and -vectors of hamiltonian
    eigenvecs_adj, eigvals, eigvecs = decompose_hamiltonian(ham)

    exp_eigvals = np.exp(-beta * eigvals)
    # Calculate greens-function

    arg = np.add.outer(exp_eigvals, exp_eigvals) / np.subtract.outer(omega + mu, eigvals)
    greens = np.einsum(subscript_str, eigenvecs_adj, arg, eigvecs)

    return greens

def btest(state, index):
    """A bit test that evaluates if 'state' in binany has a one in the 'index' location. returns one if true"""
    return (state >> index) & 1


def destruct(particles, index):
    """Fermion annihilation operator in matrix representation for a indexed
       particle in a bounded N-particles fermion fock space"""

    mat = np.zeros((2**particles, 2**particles))

    flipper = 2**index
    for i in range(2**particles):
        ispin = btest(i, index)
        if ispin == 1:
            mat[i ^ flipper, i] = phase(i, index)
    return csr_matrix(mat)

def phase(state, index):
    """Counts the number of fermions present in the state before the indexed
       one. Then returns the fermion phase sign"""
    amount = bin(state >> index+1).count('1')
    if amount % 2 == 0:
        return 1
    else:
        return -1

def destruct_operators(n):
    return [destruct(n, i) for i in range(n)]


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




def main():
    eta = 0.01j
    eps, t = 2, 1
    u = 10
    temp = 5


    omegas = np.linspace(-10, 10, 1000)

    operators = destruct_operators(4)
    ham = hamiltonian(operators, u, eps, eps, t)

    eig_e, eig_states = ham.eig()
    gf_up = gf_lehmann(eig_e, eig_states, operators[0].T, 1/temp, omegas + eta)
    gf_down = gf_lehmann(eig_e, eig_states, operators[1].T, 1/temp, omegas + eta)

    plot = Plot()
    plot.plot(omegas, -gf_up.imag)
    plot.plot(omegas, -gf_down.imag)
    plot.show()





if __name__ == "__main__":
    main()
