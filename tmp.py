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
from sciutils import Plot, Colors, set_cycler
from cmpy.core import get_omegas, diagonalize, plot_greens_function
from cmpy.models import Siam
from cmpy.dmft.two_site import TwoSiteDmft


def greens_function2(ham, states, z, beta, idx=None):
    eigvals, eigvecs = diagonalize(ham)  # ham.eig()
    idx = idx if idx is not None else np.arange(len(eigvals))
    eigvecs_adj = np.conj(eigvecs).T

    ew = np.exp(-beta * eigvals)
    partition = ew[idx].sum()
    weight = np.add.outer(ew, ew)
    gap = np.add.outer(-eigvals, eigvals)

    gf = np.zeros_like(z)
    for n in idx:
        bra = states[n]
        for m in idx:
            ket = states[m]
            if bra == ket.annihilate(0):
                gf += np.dot(eigvecs_adj[n], eigvecs[m])**2 / (z - gap[n, m]) * weight[n, m]
    return gf / partition


def main():
    u = 2
    t = 1
    mu = u/2
    beta = 1/100
    eps_imp, eps_bath, v = 0, mu, 1
    omegas, eta = get_omegas(6, 1000, 0.5)
    z = omegas + eta
    # ----------------------------------------------
    solver = TwoSiteDmft(z, u, eps_imp, v)
    solver.solve_self_consistent()
    print(f"Self-consistency reached: {solver.param_str()}")
    print()
    solver.solve()
    gf = solver.gf_latt
    Plot.quickplot(omegas, -gf.imag)



if __name__ == "__main__":
    main()
