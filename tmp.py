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
from cmpy import Hamiltonian, gf_lehmann
from cmpy.models import HubbardModel, TightBinding, Siam
from scipy.sparse import csr_matrix
import colorcet as cc

use_cycler()

class State:

    UP_CHAR = r"u"
    DN_CHAR = r"d"

    def __init__(self, array, n_spins=1):
        array = np.asarray(array, dtype="int")
        self.n_spins = n_spins
        self.n = array.shape[0]
        self.n_particles = int(self.n / self.n_spins)
        self.array = array

    @classmethod
    def spinarrays(cls, arrays):
        n_spins = len(arrays)
        array = np.array(arrays).T
        n_part = array.shape[0]
        n = n_spins * n_part
        return cls(array.reshape(n), n_spins)

    @property
    def particles(self):
        return np.sum(self.array)

    def annihilate(self, i):
        state = None
        if self.array[i] == 1:
            array = self.array.copy()
            array[i] = 0
            state = State(array, self.n_spins)
        return state

    def create(self, i):
        state = None
        if self.array[i] == 0:
            array = self.array[:]
            array[i] = 1
            state = State(array, self.n_spins)
        return state

    def _get_sitechar(self, i, latex=False):
        i0 = self.n_spins * i
        u, d = self.array[i0:i0+2]
        if u and d:
            return "d"
        elif u and not d:
            return self.UP_CHAR if not latex else r"$\uparrow$"
        elif not u and d:
            return self.DN_CHAR if not latex else r"$\downarrow$"
        return "."

    def label(self, latex=False):
        return " ".join([self._get_sitechar(i, latex) for i in range(self.n_particles)])

    def __repr__(self):
        return self.label()

    def __getitem__(self, item):
        return self.array[item]

    def __eq__(self, other_state):
        return np.all(self.array == other_state.array)

    def index(self, i, s=0):
        return i * self.n_spins + s


def greens(ham, indices, omegas, mu=0., beta=0.):
    eigvals, eigvecs = la.eigh(ham)
    eigvecs_adj = np.conj(eigvecs).T
    n = len(eigvals)
    ew = np.exp(-beta * eigvals)
    weights = np.add.outer(ew, ew)
    z = ew.sum()

    arg = np.subtract.outer(omegas + mu, eigvals)

    gf = np.zeros_like(omegas)
    for i in indices:
        gf += weights[i, i]/arg[:, i]
    return gf / z


def destruct_operators(n):
    return [destruct(n, i) for i in range(n)]


def phase(state, index):
    """Counts the number of fermions present in the state before the indexed
       one. Then returns the fermion phase sign"""
    amount = bin(state >> index+1).count('1')
    return 1 if amount % 2 == 0 else -1


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
            mat[i ^ flipper, i] = phase(i, index)
    return csr_matrix(mat)


def hamiltonian(operators, u, eps_imp, eps, v):
    d_up, d_dw, c_up, c_dw = operators
    ham = eps_imp * (d_up.T * d_up + d_dw.T * d_dw)
    ham += u * (d_up.T * d_up * d_dw.T * d_dw)
    ham += eps * (c_up.T * c_up + c_dw.T * c_dw)
    ham += v * np.abs(d_up.T * c_up + d_dw.T * c_dw + c_up.T * d_up + c_dw.T * d_dw)
    return Hamiltonian(ham.todense())


def diagonalize(operator):
    """diagonalizes single site Spin Hamiltonian"""
    eig_values, eig_vecs = la.eigh(operator)
    emin = np.amin(eig_values)
    eig_values -= emin

    return eig_values, eig_vecs

def gf_lehmann(ham, d_dag, beta, omega):
    """Outputs the lehmann representation of the greens function
       omega has to be given, as matsubara or real frequencies"""

    eigvals, eigstates = diagonalize(ham)
    ew = np.exp(-beta*eigvals)
    zet = ew.sum()
    G = np.zeros_like(omega)
    basis_create = np.dot(eigstates.T, d_dag.dot(eigstates))
    tmat = np.square(basis_create)

    tmat = tmat * np.add.outer(ew, ew)
    gap = np.add.outer(-eigvals, eigvals)

    N = eigvals.size
    for i, j in product(range(N), range(N)):
        G += tmat[i, j] / (omega + gap[i, j])
    return G / zet


def subsection(array, indices):
    idx = indices[:, None], indices
    return array[idx]


def imp_free_gf(omegas, e_c, hyb, mu=0.):
    """Outputs the Green's Function of the free propagator
    of the impurity"""
    hyb2 = hyb**2
    return (omegas - e_c + mu) / ((omegas + mu) * (omegas - e_c + mu) - hyb2)


def main():
    eta = 0.01j
    temp = 5
    beta = 1 / temp
    omegas = np.linspace(-15, 15, 1000)

    siam = Siam(1, 10, 3, 1)
    siam.sort_states([5, 3, 2, 0, 4, 1])

    arr = np.arange(16, dtype="uint8")[:, np.newaxis]
    states = [State(x[4:], n_spins=2) for x in np.unpackbits(arr, axis=1)]

    operators = destruct_operators(4)


    idx = np.array([i for i in range(16) if states[i].particles == 2])
    states = [states[i] for i in idx]

    ham = hamiltonian(operators, 2, 0, 0, 1)

    # operators = [subsection(o, idx) for o in operators]
    # ham = subsection(ham, idx)
    # ham.show(basis_labels=[s.label(True) for s in states])
    gf_0 = imp_free_gf(omegas, 0, 1)
    gf_u = gf_lehmann(ham, operators[1].T, beta, omegas)
    sigma = 1/gf_0 - 1/gf_u
    # gf_d = gf_lehmann(ham, operators[1].T, beta, omegas + eta)

    plot = Plot()
    plot.plot(omegas, gf_u)
    plot.plot(omegas, gf_0)
    plot.plot(omegas, sigma)
    # plot.plot(omegas, -gf_d.imag)
    plot.show()


if __name__ == "__main__":
    main()
