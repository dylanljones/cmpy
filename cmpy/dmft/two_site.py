# -*- coding: utf-8 -*-
"""
Created on 12 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from scipy import integrate
from .utils import self_energy, mix_values, bethe_gf_omega, bethe_dos, ErrorStats


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


def run_twosite_dmft(z, model, siam, beta=1., mixing=1.0, vtol=1e-5, nmax=1000, ref=True):
    v = siam.v + 1e-10
    m2 = model.t ** 2

    stats = ErrorStats()
    sigma = 0

    done = False
    delta_v, delta_n = 0.0, 0.0
    for it in range(nmax):
        # Update parameters of SIAM
        siam.update_hybridization(v)

        # Solve impurity problem to obtain the self energy
        if ref:
            gf_imp = impurity_gf_ref(z, siam.u[0], siam.v[0])
        else:
            gf_imp = siam.impurity_gf(z, beta)
        gf_imp0 = siam.impurity_gf_0(z)
        sigma = self_energy(gf_imp0, gf_imp)

        # Compute new hybridization
        qp_weight = quasiparticle_weight(z.real, sigma)
        v_new = np.sqrt(qp_weight * m2)
        v_new = mix_values(v, v_new, mixing)

        if done:
            break
        # Set done-flag if convergence of hybridization is reached
        delta_v = np.linalg.norm(v - v_new)
        stats.update(delta_v, delta_n)
        v = v_new
        if qp_weight == 0 or delta_v < vtol:
            done = True

    # Compute and plot lattice Green's function
    gf_latt = bethe_gf_omega(z + model.mu - sigma, model.t)
    return gf_latt, stats
