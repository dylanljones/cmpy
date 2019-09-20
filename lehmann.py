# -*- coding: utf-8 -*-
"""
Created on 17 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
import scipy.linalg as la
from scipy import integrate
from itertools import product
from sciutils import Plot, use_cycler, adj
from cmpy import get_omegas, annihilate, expectation, phase
from cmpy.models import Siam

use_cycler()


def self_energy(gf_imp0, gf_imp):
    """ Calculate the self energy from the non-interacting and interacting Green's function"""
    return 1/gf_imp0 - 1/gf_imp


def diagonalize(operator):
    eigvals, eigvecs = la.eigh(operator)
    eigvals -= np.amin(eigvals)
    return eigvals, eigvecs


def greens_function(eigvals, eigstates, c, z, beta=1.):
    basis = np.dot(eigstates.T, c.dot(eigstates))
    qmat = np.square(basis)
    # Calculate the energy gap matrix
    gap = np.add.outer(-eigvals, eigvals)
    # Calculate weights and partition function
    ew = np.exp(-beta*eigvals)
    weights = np.add.outer(ew, ew)
    partition = ew.sum()
    # Construct Green's function
    n = eigvals.size
    gf = np.zeros_like(z)
    for i, j in product(range(n), range(n)):
        gf += qmat[i, j] / (z - gap[i, j]) * weights[i, j]
    return gf / partition


def main():
    u = 5
    mu = u/2
    eps, t = 0, 1
    beta = 1/100
    omegas, eta = get_omegas(5, deta=0.01)
    z = omegas + eta

    siam = Siam(u, eps, mu, t, mu, beta)
    c = siam.c_up[0]
    # siam.set_basis([1, 2])
    # siam.sort_states(key=lambda x: x.particles)
    ham = siam.hamiltonian()

    eigvals, eigstates = diagonalize(ham)
    gf0 = siam.impurity_gf_free(z)
    gf = greens_function(eigvals, eigstates, c.todense(), z + mu, beta)

    plot = Plot()
    plot.plot(omegas, -gf.imag)
    plot.plot(omegas, -gf0.imag)
    plot.plot(omegas, -self_energy(gf0, gf).imag)
    plot.show()


if __name__ == "__main__":
    main()
