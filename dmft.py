# -*- coding: utf-8 -*-
"""
Created on 29 Mar 2019
author: Dylan

project: cmpy2
version: 1.0
"""
import numpy as np
from scipy import integrate
from scitools import Plot
from cmpy import spectral, gf_lehmann, bethe_gf_omega
from cmpy.dmft import solvers, quasiparticle_weight



def bethe_dos(energy, hopping):
    """Bethe lattice in inf dim density of states"""
    energy = np.asarray(energy).clip(-2*hopping, 2*hopping)
    return np.sqrt(4*hopping**2 - energy**2) / (2*np.pi*hopping**2)


def siam(u, eps0, eps1, v):
    g11 = np.eye(4) * (eps0 + eps1)
    g12 = np.zeros((4, 2))
    g12[1:3, :] = v
    g22 = np.array([[2 * eps0 + u, 0], [0, 2*eps1]])
    return np.block([[g11, g12], [g12.T, g22]])


def dmft(u, eps, t, mu, omegas):
    n = len(omegas)
    eta = 0.01j
    sigma = 0
    z = 0

    eps0 = eps
    # Guess initial parameters
    eps1 = mu
    v = t
    for i in range(100):
        print(v, eps1)
        print(f"V={v:.4f}, eps={eps1:.4f}")
        # Define impurity model and solve
        ham = siam(u, eps0, eps1, v)
        gf_imp = np.sum(gf_lehmann(ham, omegas+eta, mu=mu), axis=1)

        # Calculate self-energy
        ham_0 = siam(0, eps0, eps1, v)
        gf_imp0 = np.sum(gf_lehmann(np.array([[eps0, v], [v, eps1]]), omegas+eta, mu=mu), axis=1)
        sigma = (1/gf_imp0 - 1/gf_imp).real
        # Calculate impurity filling
        n_imp = filling(omegas, gf_imp)
        print(f"Impurity filling: {n_imp:.4f}")

        # Calculate quasiparticle weight
        idx = int(n/2)
        dsigma = np.gradient(sigma)
        z = 1/(1-dsigma[idx])
        print(f"Quasi-particle weight: {z:.4f}")
        # Calculate lattice Green's function
        gf_latt = bethe_gf_omega(omegas + eta + eps0 + mu - sigma)
        n_latt = filling(omegas, gf_latt)
        print(f"Lattice filling: {n_latt:.4f}")

        # Update parameters
        v = np.sqrt(z * t**2)
        delta = abs(n_imp - n_latt)
        print(f"DELTA: {delta}\n")

    plot = Plot()
    plot.plot(omegas, sigma.real, label="RE")
    plot.plot(omegas, sigma.imag, label="IM")
    plot.legend()
    plot.show()


class TwoSiteSiam:

    def __init__(self, u, eps0, eps1, v, mu):
        self.mu = mu
        self.ham0 = np.array([[eps0, v], [v, eps1]])

        g11 = np.eye(4) * (eps0 + eps1)
        g12 = np.zeros((4, 2))
        g12[1:3, :] = v
        g22 = np.array([[2 * eps0 + u, 0], [0, 2*eps1]])
        self.ham = np.block([[g11, g12], [g12.T, g22]])

    def gf_imp0(self, omegas):
        return gf_lehmann(self.ham0, omegas, mu=self.mu)[:, 0]

    def gf_imp(self, omegas):
        gf = gf_lehmann(self.ham, omegas, mu=self.mu)
        idx = np.array([1, 2, 3, 4])
        return 0.5 * np.sum(gf[:, idx], axis=1) / 4 + gf[:, 5]


def gf_imp_free_potthof(eps0, eps1, v, omegas, mu=0.):
    e = (eps0 - eps1) / 2
    r = np.sqrt(e*e + v*v)
    p1 = (r + e) / (omegas + mu - e - r)
    p2 = (r - e) / (omegas + mu - e + r)
    return (p1 + p2) / (2 * r)


def filling(omegas, gf):
    idx = np.argmin(np.abs(omegas)) + 1
    return -2 * integrate.trapz(gf[:idx].imag, omegas[:idx]) / np.pi


def sigma_potthof(u, v, omegas):
    return u/2 + u**2/8 * (1/(omegas - 3*v) + 1 / (omegas + 3 * v))


def dos(gf):
    return -1/np.pi * gf.imag


def main():
    eta = 0.01j
    u, eps, t = 5, 0, 1
    mu = u / 2
    n = 100000
    dw = 20

    eps0, eps1 = eps, mu
    v = 1

    omegas = np.linspace(-dw, dw, n + 1)
    # dmft(u, eps, t, mu, omegas)
    siam = TwoSiteSiam(u, eps0, eps1, v, mu)

    gf_imp0 = siam.gf_imp0(omegas + eta)
    gf_imp0_potthoff = gf_imp_free_potthof(eps0, eps1, v, omegas + eta, mu)

    plot = Plot()
    plot.plot(omegas, dos(gf_imp0))
    plot.plot(omegas, dos(gf_imp0_potthoff))
    plot.show()

    return
    gf_imp0 = siam.gf_imp0(1j * omegas)
    gf_imp = siam.gf_imp(1j * omegas)
    sigma = (1/gf_imp0 - 1/gf_imp)

    n_imp = filling(omegas, siam.gf_imp(omegas + eta))
    print(n_imp)

    idx = np.argmin(np.abs(omegas))
    dsigma = np.gradient(sigma)
    z = 1 / (1 - dsigma[idx].real)
    v = np.sqrt(z * t ** 2)
    print(v)

    print(z)

    gf_latt = bethe_gf_omega(omegas + eta + mu - sigma.imag, 2)
    idx = np.argmin(np.abs(omegas))
    dsigma = np.gradient(sigma)
    z = 1 / (1 - dsigma[idx].real)
    print(f"Z={z}")

    plot = Plot(ylim=(-1, 2))
    plot.plot(omegas, dos(gf_imp))
    # plot.plot(omegas, sigma.imag)
    #plot.plot(omegas, dos(gf_latt), color="r")
    plot.show()


if __name__ == "__main__":
    main()
