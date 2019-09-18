# -*- coding: utf-8 -*-
"""
Created on 17 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
import scipy.linalg as la
from itertools import product
from sciutils import Plot, use_cycler
from cmpy import get_omegas, annihilate
from cmpy.models import Siam

use_cycler()


def diagonalize(operator):
    eig_values, eig_vecs = la.eigh(operator)
    emin = np.amin(eig_values)
    eig_values -= emin
    return eig_values, eig_vecs


def greens_function(ham, c, z, beta=1.):
    """Outputs the lehmann representation of the greens function
       omega has to be given, as matsubara or real frequencies"""
    c = c.dense

    eigvals, eigstates = diagonalize(ham)  # la.eigh(ham)
    ew = np.exp(-beta*eigvals)
    partition = ew.sum()

    basis = np.dot(eigstates.T, c.dot(eigstates))
    tmat = np.square(basis)
    gap = np.add.outer(-eigvals, eigvals)
    weights = np.add.outer(ew, ew)

    n = eigvals.size
    gf = np.zeros_like(z)
    for i, j in product(range(n), range(n)):
        gf += tmat[i, j] / (z - gap[i, j]) * weights[j, i]
    return gf / partition


def greens_function2(ham, states, z, beta, spin=0):
    eigvals, eigstates = diagonalize(ham)  # la.eigh(ham)
    eigstates_adj = np.conj(eigstates).T
    ew = np.exp(-beta*eigvals)
    partition = ew.sum()

    tmat = np.square(np.inner(eigstates_adj, eigstates))
    gap = np.add.outer(-eigvals, eigvals)
    weights = np.add.outer(ew, ew)

    n = len(states)
    gf = np.zeros_like(z)
    for j in range(n):
        ket = states[j]
        other = annihilate(ket, spin)
        if other is not None:
            try:
                i = states.index(other)
                gf += tmat[i, j] / (z - gap[i, j]) * weights[i, j]
            except ValueError:
                pass
    return gf / partition


def main():
    u = 5
    mu = u/2
    eps, t = 0, 1
    beta = 1/100
    omegas, eta = get_omegas(8)
    z = omegas + eta

    siam = Siam(u, eps, mu, t, mu, beta)
    # siam.set_basis([1, 2])
    # siam.sort_states(key=lambda x: x.particles)
    ham = siam.hamiltonian()
    ham.show(False, show_values=True, labels=siam.state_labels())
    # siam.show_hamiltonian(False)

    gf = greens_function(ham, siam.c_up[0], z + mu, beta)
    gf2 = greens_function2(ham, siam.states, z + mu, beta)

    plot = Plot()
    plot.plot(omegas, -gf.imag, label=r"G")
    plot.plot(omegas, -gf2.imag, label=r"G$_2$")
    plot.legend()
    plot.show()


if __name__ == "__main__":
    main()
