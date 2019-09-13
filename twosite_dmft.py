# -*- coding: utf-8 -*-
"""
Created on 11 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
import scipy.linalg as la
from scipy import integrate
from sciutils import Plot, Colors, Linestyles, use_cycler
from cmpy import bethe_gf_omega, Hamiltonian, gf_lehmann
from cmpy.models import Siam, HubbardModel
from itertools import product

use_cycler()

def dos(gf):
    return -1/np.pi * gf.imag


def bethe_dos(z, t):
    """Bethe lattice in inf dim density of states"""
    energy = np.asarray(z).clip(-2 * t, 2 * t)
    return np.sqrt(4 * t**2 - energy**2) / (2 * np.pi * t**2)


# ========================== REFERENCES =======================================


def potthoff_gf_imp0_original(eps0, eps1, v, omegas, mu=0.):
    e = (eps0 - eps1) / 2
    r = np.sqrt(e*e + v*v)
    p1 = (r + e) / (omegas + mu - e - r)
    p2 = (r - e) / (omegas + mu - e + r)
    return (p1 + p2) / (2 * r)


def potthoff_gf_imp0(eps0, eps1, v, omegas, mu=0.):
    e = (eps1 - eps0) / 2
    r = np.sqrt(e*e + v*v)
    p1 = (r - e) / (omegas + mu - e - r)
    p2 = (r + e) / (omegas + mu - e + r)
    return (p1 + p2) / (2 * r)


def potthoff_sigma(u, v, omegas):
    return u/2 + u**2/8 * (1/(omegas - 3*v) + 1 / (omegas + 3 * v))

# =============================================================================

def decompose_hamiltonian(ham):
    h, ev = la.eigh(ham)
    ev_inv = ev.conj().T
    return ev_inv, h, ev

def m2_weight(t):
    return integrate.quad(lambda x: x*x * bethe_dos(x, t), -2*t, 2*t)[0]


def quasiparticle_weight(omegas, sigma):
    idx = np.argmin(np.abs(omegas))
    dsigma = sigma[idx] - sigma[idx-1]
    domega = omegas[idx] - sigma[idx-1]
    return 1/(1 - dsigma/domega)


def ham_siam(u, eps0, eps1, v):
    g11 = np.eye(4) * (eps0 + eps1)
    g12 = np.zeros((4, 2))
    g12[1:3, :] = v
    g22 = np.array([[2 * eps0 + u, 0], [0, 2*eps1]])
    return np.block([[g11, g12], [g12.T, g22]])


def ham0_siam(eps0, eps1, v):
    ham = Hamiltonian.zeros(2)
    ham.fill_diag([eps0, eps1])
    ham.fill_diag(v, 1)
    ham.fill_diag(v, -1)
    return ham


def filling(omegas, gf):
    idx = np.argmin(np.abs(omegas)) + 1
    return -2 * integrate.trapz(gf[:idx].imag, omegas[:idx]) / np.pi


def test_noninteracting_gf(eps_imp, u, eps_bath, v, mu, omegas):
    siam = Siam(eps_imp, u, eps_bath, v, mu=mu)
    siam.sort_states([5, 3, 2, 0, 4, 1])
    gf_imp0_potthoff_original = potthoff_gf_imp0_original(eps_imp, eps_bath, v, omegas, mu=mu)
    gf_imp0_potthoff = potthoff_gf_imp0(eps_imp, eps_bath, v, omegas, mu=mu)
    gf_imp0 = siam.gf_imp_free(omegas).T[0]
    plot = Plot()
    plot.plot(omegas.real, dos(gf_imp0), color="k", lw=2)
    plot.plot(omegas.real, dos(gf_imp0_potthoff), color="C0", ls="--")
    plot.plot(omegas.real, dos(gf_imp0_potthoff_original), color="r", lw=0.5)
    plot.show()


def main():
    # lattice and other parameters
    eta = 0.01j
    eps, t = 0, 1
    u = 1
    mu = u/2

    omax = 10
    omegas = np.linspace(-omax, omax, 10000)

    # Impurity model
    eps_imp = eps
    eps_bath = mu
    v = t

    siam = Siam(eps_imp, u, eps_bath, v, mu=mu)
    siam.sort_states([5, 3, 2, 0, 4, 1])
    # ===========================================
    # test_noninteracting_gf(eps_imp, u, eps_bath, v, mu, omegas + eta)

    gf_imp0 = siam.gf_imp_free(omegas + eta).T[0]
    # gf_imp = impurity_gf(siam.hamiltonian(), omegas + eta)
    # gf_imp2 = gf_lehmann(siam.hamiltonian(), omegas + eta).sum(axis=1)

    plot = Plot()
    plot.plot(omegas, -gf_imp0.imag)

    # plot.plot(omegas, -gf_imp.imag)


    # plot.plot(omegas, -gf_imp2.imag, Colors.bgreen)
    plot.show()



if __name__ == "__main__":
    main()
