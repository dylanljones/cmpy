# -*- coding: utf-8 -*-
"""
Created on 12 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from scipy import integrate
from cmpy import Siam, bethe_dos, bethe_gf_omega, self_energy, get_eta

# ========================== REFERENCES =======================================


def potthoff_gf_imp0_original(eps_imp, eps_bath, v, omegas, mu=0.):
    e = (eps_imp - eps_bath) / 2
    r = np.sqrt(e*e + v*v)
    p1 = (r + e) / (omegas + mu - e - r)
    p2 = (r - e) / (omegas + mu - e + r)
    return (p1 + p2) / (2 * r)


def potthoff_gf_imp0(eps_imp, eps_bath, v, omegas, mu=0.):
    e = (eps_bath - eps_imp) / 2
    r = np.sqrt(e*e + v*v)
    p1 = (r - e) / (omegas + mu - e - r)
    p2 = (r + e) / (omegas + mu - e + r)
    return (p1 + p2) / (2 * r)


def potthoff_sigma(u, v, omegas):
    return u/2 + u**2/8 * (1/(omegas - 3*v) + 1/(omegas + 3*v))

# =============================================================================


def filling(omegas, gf):
    idx = np.argmin(np.abs(omegas)) + 1
    return -2 * integrate.trapz(gf[:idx].imag, omegas[:idx]) / np.pi


def m2_weight(t):
    return integrate.quad(lambda x: x*x * bethe_dos(x, t), -2*t, 2*t)[0]


def quasiparticle_weight(omegas, sigma):
    dw = omegas[1] - omegas[0]
    win = (-dw <= omegas) * (omegas <= dw)
    dsigma = np.polyfit(omegas[win], sigma.real[win], 1)[0]
    z = 1/(1 - dsigma)
    return max(0, z)


class TwoSiteDmft:

    def __init__(self, omega, u=5, eps=0, v=1, mu=None, eps_bath=None, beta=10., deta=1):
        mu = mu or u/2
        eps_bath = eps_bath or mu
        self.siam = Siam(u, eps, eps_bath, v, mu, beta)
        self.z = omega + get_eta(omega, deta)
        self.gf_imp0 = None
        self.gf_imp = None
        self.sigma = None

    @property
    def u(self):
        return self.siam.u

    @property
    def eps_imp(self):
        return self.siam.eps_imp

    @property
    def eps_bath(self):
        return self.siam.eps_bath[0]

    @property
    def v(self):
        return self.siam.v[0]

    @property
    def mu(self):
        return self.siam.mu

    def update_bath_energy(self, eps_bath):
        self.siam.update_bath_energy(eps_bath)

    def update_hopping(self, v):
        self.siam.update_hopping(v)

    def update_bath(self, eps_bath, v):
        self.siam.update_bath(eps_bath, v)

    def solve(self):
        self.gf_imp0 = self.siam.impurity_gf_free(self.z)
        self.gf_imp = self.siam.impurity_gf(self.z)
        self.sigma = self_energy(self.gf_imp0, self.gf_imp)

    def dos_imp(self):
        return -self.gf_imp.imag

    def dos_imp0(self):
        return -self.gf_imp0.imag

    def gf_latt(self):
        return bethe_gf_omega(self.z + self.mu - self.sigma, 2*self.v)

    def dos_latt(self):
        return -self.gf_latt().imag

    def m2_weight(self):
        return m2_weight(self.v)

    def quasiparticle_weight(self):
        return quasiparticle_weight(self.z.real, self.sigma)
