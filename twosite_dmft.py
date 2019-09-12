# -*- coding: utf-8 -*-
"""
Created on 11 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from scipy import integrate
from sciutils import Plot
from cmpy import gf_lehmann
from cmpy.models import Siam, HubbardModel


def bethe_dos(z, t):
    """Bethe lattice in inf dim density of states"""
    energy = np.asarray(z).clip(-2 * t, 2 * t)
    return np.sqrt(4 * t**2 - energy**2) / (2 * np.pi * t**2)


def m2_weight(t):
    return integrate.quad(lambda x: x*x * bethe_dos(x, t), -2*t, 2*t)[0]


def diagonalize(operator):
    """diagonalizes single site Spin Hamiltonian"""
    eig_values, eig_vecs = np.linalg.eigh(operator)
    emin = np.amin(eig_values)
    eig_values -= emin
    return eig_values, eig_vecs


def gf_imp_free_potthof(eps0, eps1, v, omegas, mu=0.):
    e = (eps0 - eps1) / 2
    r = np.sqrt(e*e + v*v)
    p1 = (r + e) / (omegas + mu - e - r)
    p2 = (r - e) / (omegas + mu - e + r)
    return (p1 + p2) / (2 * r)


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


def filling(omegas, gf):
    idx = np.argmin(np.abs(omegas)) + 1
    return -2 * integrate.trapz(gf[:idx].imag, omegas[:idx]) / np.pi


def sigma_potthof(u, v, omegas):
    return u/2 + u**2/8 * (1/(omegas - 3*v) + 1 / (omegas + 3 * v))


def dos(gf):
    return -1/np.pi * gf.imag


def main():
    eta = 0.01j
    u = 5
    eps, t = 0, 1
    mu = u/2

    siam = Siam(eps_imp=eps, u=u, eps=mu, v=t, mu=mu)
    siam.sort_states([5, 3, 2, 0, 4, 1])
    # siam.show_hamiltonian()
    omax = 10
    omegas = np.linspace(-omax, omax, 1000)
    ham = siam.hamiltonian()

    gf = gf_lehmann(ham, omegas + eta, mu=mu).sum(axis=1) / 4
    print(filling(omegas, gf))

    plot = Plot()
    plot.plot(omegas, - gf.imag / np.pi)
    plot.show()

    return
    gf_imp0 = gf_imp_free_potthof(eps0, eps1, v, omegas + eta, mu)
    sigma = sigma_potthof(u, v, omegas)
    dos = bethe_dos(omegas + mu + sigma, 1)
    # xi = omegas + eta + mu + eps - sigma
    # gf = bethe_gf_omega(xi, 2)

    # n_latt = filling(omegas, gf)
    # print(f"n_latt: {n_latt}")

    # z = 2 * 18 * v * v / (u * u)    # quasiparticle_weight(omegas, sigma)
    # print(f"Quasiparticle-weight: {z}")


if __name__ == "__main__":
    main()
