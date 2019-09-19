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
from cmpy import get_omegas, annihilate, expectation
from cmpy.models import Siam

use_cycler()


def diagonalize(operator):
    eigvals, eigvecs = la.eig(operator)
    eigvals -= np.amin(eigvals)
    idx = np.argsort(eigvals)
    return eigvals, eigvecs


def greens_function(eigvals, eigstates, c, z, beta=1.):
    """Outputs the lehmann representation of the greens function
       omega has to be given, as matsubara or real frequencies"""
    c = c.dense

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


def expectation(eigvals, eigstates, operator, beta):
    operator = operator.todense()
    ew = np.exp(-beta * eigvals)
    aux = np.einsum('i,ji,ji', ew, eigstates, operator.dot(eigstates))
    return aux / ew.sum()


def main():
    u = 5
    mu = u/2
    eps, t = 0, 1
    beta = 1/100
    omegas, eta = get_omegas(10, deta=0.1)
    z = omegas + eta

    siam = Siam(u, eps, mu, t, mu, beta)
    c = siam.c_up[0]
    # siam.set_basis([1, 2])
    # siam.sort_states(key=lambda x: x.particles)
    ham = siam.hamiltonian()

    eigvals, eigstates = diagonalize(ham)
    gf = greens_function(eigvals, eigstates, c, z + mu, beta)


    print(expectation(eigvals, eigstates, c.T * c, beta))

    Plot.quickplot(omegas, -gf.imag)




if __name__ == "__main__":
    main()
