# -*- coding: utf-8 -*-
"""
Created on 12 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from scipy import integrate
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

    def __init__(self, z, u=5, eps=0, t=1, mu=None, eps_bath=None, beta=10.):
        mu = mu or u/2
        eps_bath = eps_bath or mu
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

    # =========================================================================

    def solve(self):
        self.gf_imp0 = self.siam.impurity_gf_free(self.z)
        self.gf_imp = self.siam.impurity_gf(self.z)
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

    def solve_self_consistent(self, thresh=1e-4, mixing=0.0, verbose=True, inline=True, nmax=10000,
                              header=""):
        cout = Terminal(enabled=verbose)
        if inline:
            cout.write()
            writer = cout.updateln
        else:
            writer = cout.writeln
        slen = len(str(nmax-1))

        v = self.v + 0.1
        for i in range(nmax):
            self.update_hybridization(v)
            self.solve()
            v_new = self.new_hybridization(mixing)
            delta = abs(v - v_new)
            v = v_new

            idxstr = f"[{i}]"
            writer(header + f"{idxstr:<{slen+2}} v={float(v):.4f} (delta={float(delta):.2e})")
            if delta <= thresh:
                break
        self.update_hybridization(v)

        if i == (nmax - 1):
            writer(header + f"Aborted: maximal iteration {nmax} reached (delta={float(delta):.2e})")
        else:
            writer(header + f"Threshold of {thresh:.1e} reached (iter: {i}, delta={float(delta):.2e})")
        if inline:
            cout.writeln()
        return delta
