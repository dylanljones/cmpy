# -*- coding: utf-8 -*-
"""
Created on 12 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from scipy import integrate
from scipy import optimize
from sciutils import Terminal
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
    x = omegas[:idx]
    y = -gf[:idx].imag
    x[-1] = 0
    y[-1] = (y[-1] + y[-2]) / 2
    return integrate.simps(y, x)


def m2_weight(t):
    return integrate.quad(lambda x: x*x * bethe_dos(x, t), -2*t, 2*t)[0]


def quasiparticle_weight(omegas, sigma):
    dw = omegas[1] - omegas[0]
    win = (-dw <= omegas) * (omegas <= dw)
    dsigma = np.polyfit(omegas[win], sigma.real[win], 1)[0]
    z = 1/(1 - dsigma)
    if z < 0.01:
        z = 0
    return max(0, z)


class TwoSiteDmft:

    def __init__(self, z, u=5, eps=0, t=1, mu=None, eps_bath=None, beta=10.):
        mu = u / 2 if mu is None else mu
        eps_bath = mu if eps_bath is None else eps_bath
        self.z = z
        self.t = t
        self.m2 = m2_weight(t)

        self.siam = Siam(u, eps, eps_bath, t, mu, beta)
        self.gf_imp0 = None
        self.gf_imp = None
        self.sigma = None
        self.gf_latt = None
        self.quasiparticle_weight = None

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

    @property
    def omega(self):
        return self.z.real

    def update_bath_energy(self, eps_bath):
        self.siam.update_bath_energy(eps_bath)

    def update_hybridization(self, v):
        self.siam.update_hybridization(v)

    def update_bath(self, eps_bath, v):
        self.siam.update_bath(eps_bath, v)

    def param_str(self, dec=2):
        u = f"u={self.u:.{dec}}"
        eps_imp = f"eps_imp={self.eps_imp:.{dec}}"
        eps_bath = f"eps_bath={self.eps_bath:.{dec}}"
        v = f"v={self.v:.{dec}}"
        return ", ".join([u, eps_imp, eps_bath, v])

    # =========================================================================

    def solve(self, spin=0):
        self.gf_imp0 = self.siam.impurity_gf_free(self.z)
        self.gf_imp = self.siam.impurity_gf(self.z, spin=spin)
        self.sigma = self_energy(self.gf_imp0, self.gf_imp)
        self.gf_latt = bethe_gf_omega(self.z + self.mu - self.sigma, 2*self.t)
        self.quasiparticle_weight = quasiparticle_weight(self.z.real, self.sigma)

    def new_hybridization(self, mixing=0.0):
        z = self.quasiparticle_weight
        v_new = np.sqrt(z * self.m2)
        if mixing:
            new, old = 1 - mixing, mixing
            v_new = new * v_new + old * self.v
        return v_new

    def impurity_filling(self):
        # c = self.siam.c_up[0]
        return filling(self.omega, self.gf_imp) / np.pi

    def lattice_filling(self):
        return filling(self.omega, self.gf_latt) / np.pi

    def filling_condition(self, eps_bath):
        self.update_bath_energy(eps_bath)
        self.solve()
        return self.impurity_filling() - self.lattice_filling()

    def optimize_filling(self, tol=1e-2):
        sol = optimize.root(self.filling_condition, x0=self.eps_bath, tol=tol)
        if not sol.success:
            raise ValueError(f"Failed to optimize filling! ({self.param_str()})")
        else:
            self.update_bath_energy(sol.x)
            self.solve()

    def solve_self_consistent(self, spin=0, thresh=1e-4, mixing=0.0, nmax=10000):
        v = self.v + 0.1
        delta, i = 0, 0
        for i in range(nmax):
            self.optimize_filling()

            self.update_hybridization(v)
            self.solve(spin)
            if self.quasiparticle_weight == 0:
                break
            v_new = self.new_hybridization(mixing)
            delta = abs(v - v_new)
            v = v_new
            if delta <= thresh:
                break
        self.update_hybridization(v)
        return delta
