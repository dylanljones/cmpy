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
from cmpy import bethe_gf_omega, Hamiltonian, gf_lehmann
from cmpy.models import Siam, HubbardModel


def dos(gf):
    return -1/np.pi * gf.imag


def bethe_dos(z, t):
    """Bethe lattice in inf dim density of states"""
    energy = np.asarray(z).clip(-2 * t, 2 * t)
    return np.sqrt(4 * t**2 - energy**2) / (2 * np.pi * t**2)


def gf_lehmann2(ham, omega, mu=0., idx=None):
    """ Calculate the greens function of the given Hamiltonian

    Parameters
    ----------
    ham: array_like
        Hamiltonian matrix
    omega: complex or array_like
        Energy e+eta of the greens function (must be complex!)
    mu: float, default: 0
        Chemical potential of the system
    only_diag: bool, default: True
        only return diagonal elements of the greens function if True

    Returns
    -------
    greens: np.ndarray
    """
    omega = np.asarray(omega)
    # Calculate eigenvalues and -vectors of hamiltonian
    eigvals, eigvecs = np.linalg.eigh(ham)
    eigenvectors_adj = np.conj(eigvecs).T

    # Calculate greens-function
    subscript_str = "ij,...j,ji->...i"
    arg = np.subtract.outer(omega + mu, eigvals)
    greens = np.einsum(subscript_str, eigenvectors_adj, 1 / arg, eigvecs)
    return greens.T

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


def gf_lehmann2(ham, omega, mu=0., idx=None):
    """ Calculate the greens function of the given Hamiltonian

    Parameters
    ----------
    ham: array_like
        Hamiltonian matrix
    omega: complex or array_like
        Energy e+eta of the greens function (must be complex!)
    mu: float, default: 0
        Chemical potential of the system
    only_diag: bool, default: True
        only return diagonal elements of the greens function if True

    Returns
    -------
    greens: np.ndarray
    """
    omega = np.asarray(omega)
    # Calculate eigenvalues and -vectors of hamiltonian
    eigvals, eigvecs = np.linalg.eigh(ham)
    eigenvectors_adj = np.conj(eigvecs).T



    # Calculate greens-function
    subscript_str = "ij,...j,ji->...i"
    arg = np.subtract.outer(omega + mu, eigvals)
    greens = np.einsum(subscript_str, eigenvectors_adj, 1 / arg, eigvecs)
    return greens.T


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
    gf_imp = siam.gf_imp(omegas + eta).sum(axis=1) / 4

    sigma = (1/gf_imp0 - 1/gf_imp).imag

    gf_latt = bethe_gf_omega(omegas + eta + mu - sigma, 2*t)

    plot = Plot()
    # plot.plot(omegas, 1/gf_imp0.real, color="k")
    # plot.plot(omegas, 1/gf_imp.real)
    plot.plot(omegas, sigma, color="r")
    # plot.plot(omegas, dos(gf_latt))
    plot.show()

    return
    gf_imp0 = potthof_gf_imp0(eps0, eps1, v, omegas + eta, mu)
    sigma = sigma_potthof(u, v, omegas)
    n = bethe_dos(omegas + mu + sigma, 1)
    # xi = omegas + eta + mu + eps - sigma
    # gf = bethe_gf_omega(xi, 2)

    # n_latt = filling(omegas, gf)
    # print(f"n_latt: {n_latt}")

    # z = 2 * 18 * v * v / (u * u)    # quasiparticle_weight(omegas, sigma)
    # print(f"Quasiparticle-weight: {z}")


if __name__ == "__main__":
    main()
