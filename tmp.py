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
from cmpy.core import get_omegas, diagonalize
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
    u = 5
    mu = u/2
    beta = 1/100
    eps_imp, eps_bath, v = 0, mu, 1
    omegas, eta = get_omegas(6, 1000, 0.5)
    z = omegas + eta
    # ----------------------------------------------
    # siam.set_basis([1, 2])
    solver = TwoSiteDmft(z)

    solver.solve_self_consistent(0)
    gf_up = solver.gf_latt
    solver.solve_self_consistent(1)
    gf_dn = solver.gf_latt

    gf = np.asarray([gf_up, gf_dn])

    ymax = np.max(-gf.imag)
    ylim = (0, 1.1 * ymax)

    plot = Plot(create=False)
    plot.set_gridspec(2, 1)


    ax = plot.add_gridsubplot(0)
    plot.set_limits(ylim=ylim)
    plot.plotfill(omegas, -gf_up.imag)
    plot.set_labels(ylabel=r"A$_{\uparrow}$")
    plot.grid()

    plot.add_gridsubplot(1, sharex=ax)
    plot.set_limits(ylim=ylim)
    plot.invert_yaxis()
    plot.plotfill(omegas, -gf_dn.imag)
    plot.set_labels(xlabel=r"$\omega$", ylabel=r"A$_{\downarrow}$")
    plot.grid()

    for ax in plot.axs:
        try:
            ax.label_outer()
        except:
            pass

    plot.show()


if __name__ == "__main__":
    main()
