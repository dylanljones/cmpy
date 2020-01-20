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
from cmpy import bethe_dos, bethe_gf_omega
from cmpy.models import Siam


# ========================== REFERENCES =======================================

# Reference functions taken from M. Potthof:
# 'Two-site dynamical mean-field theory'
def impurity_gf_free_ref(z, eps0, eps1, v):
    e = (eps1 - eps0) / 2
    r = np.sqrt(e*e + v*v)
    term1 = (r - e) / (z - e - r)
    term2 = (r + e) / (z - e + r)
    return 1/(2*r) * (term1 + term2)


# Reference functions taken from E. Lange:
# 'Renormalized vs. unrenormalized perturbation-theoretical
# approaches to the Mott transition'
def impurity_gf_ref(z, u, v):
    sqrt16 = np.sqrt(u ** 2 + 16 * v ** 2)
    sqrt64 = np.sqrt(u ** 2 + 64 * v ** 2)
    a1 = 1/4 * (1 - (u ** 2 - 32 * v ** 2) / np.sqrt((u ** 2 + 64 * v ** 2) * (u ** 2 + 16 * v ** 2)))
    a2 = 1/2 - a1
    e1 = 1/4 * (sqrt64 - sqrt16)
    e2 = 1/4 * (sqrt64 + sqrt16)
    return (a1 / (z - e1) + a1 / (z + e1)) + (a2 / (z - e2) + a2 / (z + e2))


# =============================================================================


def self_energy(gf_imp0, gf_imp):
    """ Calculate the self energy from the non-interacting and interacting Green's function"""
    return 1/gf_imp0 - 1/gf_imp


def m2_weight(t):
    """ Calculates the second moment weight"""
    return integrate.quad(lambda x: x*x * bethe_dos(x, t), -2*t, 2*t)[0]


def quasiparticle_weight(omegas, sigma):
    """ Calculates the quasiparticle weight"""
    dw = omegas[1] - omegas[0]
    win = (-dw <= omegas) * (omegas <= dw)
    dsigma = np.polyfit(omegas[win], sigma.real[win], 1)[0]
    z = 1/(1 - dsigma)
    if z < 0.01:
        z = 0
    return z


def filling(omegas, gf):
    """ Calculate the filling using the Green's function of the corresponding model"""
    idx = np.argmin(np.abs(omegas)) + 1
    x = omegas[:idx]
    y = -gf[:idx].imag
    x[-1] = 0
    y[-1] = (y[-1] + y[-2]) / 2
    return integrate.simps(y, x)


# =========================================================================



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
        self.gf_latt = bethe_gf_omega(self.z + self.mu - self.sigma, self.t)
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
        it, delta_v = 0, 0
        for it in range(nmax):
            self.optimize_filling()
            self.update_hybridization(v)
            self.solve(spin)
            if self.quasiparticle_weight == 0:
                break
            v_new = self.new_hybridization(mixing)
            delta_v = abs(v - v_new)
            v = v_new
            if delta_v <= thresh:
                break
        self.update_hybridization(v)
        return it, delta_v
