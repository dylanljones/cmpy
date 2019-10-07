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
from scitools import Plot, Colors, set_cycler
from cmpy.core import get_omegas, diagonalize, plot_greens_function
from cmpy.models import Siam
from cmpy.dmft.two_site import TwoSiteDmft



def greens_function2(eigvals, eigstates, states, z, beta, idx=None):
    idx = idx if idx is not None else np.arange(len(eigvals))
    eigstates_adj = np.conj(eigstates).T

    ew = np.exp(-beta * eigvals)
    partition = ew[idx].sum()
    weight = np.add.outer(ew, ew)
    gap = np.add.outer(-eigvals, eigvals)

    gf = np.zeros_like(z)
    for j in idx:
        ket = states[j]
        try:
            i = states.index(ket.annihilate(0))
        except ValueError:
            pass
        else:
            gf += np.dot(eigstates_adj[i], eigstates[j])**2 / (z - gap[i, j]) * weight[i, j]
    return gf / partition


def main():
    u = 5
    eps, t = 0, 1
    mu = u/2
    beta = 1/1
    omegas, eta = get_omegas(6, 1000, 0.5)
    z = omegas + eta
    # ----------------------------------------------
    siam = Siam(u, eps, mu, t, mu=mu)

    states = siam.states
    ops = siam.ops
    ham = siam.hamiltonian()
    eigvals, eigstates = diagonalize(ham)

    gf = greens_function2(eigvals, eigstates, states, z + mu, beta)
    gf0 = siam.impurity_gf_free(z)

    plot = Plot()
    plot.plot(omegas, -gf.imag)
    plot.plot(omegas, -gf0.imag)
    plot.show()





if __name__ == "__main__":
    main()
