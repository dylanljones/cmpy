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
from cmpy import spectral, gf_lehmann
from scipy.sparse import csr_matrix
from itertools import product


def bethe_gf_omega(z, half_bandwidth=1):
    """Local Green's function of Bethe lattice for infinite Coordination number.

    Taken from gf_tools by Weh Andreas
    https://github.com/DerWeh/gftools/blob/master/gftools/__init__.py

    Parameters
    ----------
    z : complex ndarray or complex
        Green's function is evaluated at complex frequency `z`
    half_bandwidth : float
        half-bandwidth of the DOS of the Bethe lattice
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`
    Returns
    -------
    bethe_gf_omega : complex ndarray or complex
        Value of the Green's function
    """
    z_rel = z / half_bandwidth
    return 2. / half_bandwidth * z_rel * (1 - np.sqrt(1 - 1 / (z_rel * z_rel)))


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


def filling(omegas, gf):
    idx = np.argmin(np.abs(omegas)) + 1
    return -2 * integrate.trapz(gf[:idx].imag, omegas[:idx]) / np.pi


def sigma_potthof(u, v, omegas):
    return u/2 + u**2/8 * (1/(omegas - 3*v) + 1 / (omegas + 3 * v))


def dos(gf):
    return -1/np.pi * gf.imag


def main():
    eta = 0.01j
    u = 2
    eps, t = 0, 1
    mu = u/2
    eps0 = eps
    eps1 = mu
    v = 0.01
    omegas = np.linspace(-20, 20, 1000)

    gf_imp0 = gf_imp_free_potthof(eps0, eps1, v, omegas + eta, mu)
    sigma = sigma_potthof(u, v, omegas)
    dos = bethe_dos(omegas + mu + sigma, 1)
    # xi = omegas + eta + mu + eps - sigma
    # gf = bethe_gf_omega(xi, 2)

    # n_latt = filling(omegas, gf)
    # print(f"n_latt: {n_latt}")

    # z = 2 * 18 * v * v / (u * u)    # quasiparticle_weight(omegas, sigma)
    # print(f"Quasiparticle-weight: {z}")

    us = np.linspace(0, 20, 1000)
    qp_weight = 1 / (1 + us*us/(v*36))
    plot = Plot()
    # plot.plot(omegas, sigma)
    plot.plot(us, qp_weight)
    plot.show()


if __name__ == "__main__":
    main()
