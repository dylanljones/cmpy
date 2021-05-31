# coding: utf-8
#
# This code is part of cmpy.
# 
# Copyright (c) 2021, Dylan Jones

import logging
import numpy as np
import matplotlib.pyplot as plt
from cmpy.models import SingleImpurityAndersonModel
from cmpy import exactdiag as ed
from cmpy.dmft import self_energy
from cmpy.dmft.twosite import (
    quasiparticle_weight, mix_values, compute_lattice_greens_function, impurity_gf_ref
)

logger = logging.getLogger("cmpy")
logger.setLevel(logging.INFO)


def compute_impurity_gf(z, siam, beta=np.inf, ref=False):
    if ref:
        return impurity_gf_ref(z, siam.u, siam.v)
    return ed.greens_function_lehmann(siam, z, beta).gf


def twosite_dmft_half_filling(z, u, t=1., beta=np.inf, mixing=0.5, vtol=1e-6, max_iter=10000,
                              vthresh=1e-10, ref=True):

    # Map the lattice model to the single impurity Anderson model at half filling
    v = [t]

    siam = SingleImpurityAndersonModel(u, v=v, mu=None, temp=1/beta)
    # Initial hybridization value must be different
    # from the current one or the error will be zero.
    v = siam.v[0] + 0.1
    # prepare m2 weight
    m2 = t ** 2

    converged = False
    it = 0
    delta_v = 0
    for it in range(max_iter):
        # Update parameters of SIAM
        siam.update_hybridization(v)

        # Solve impurity problem to obtain the self energy
        gf_imp0 = siam.impurity_gf0(z)
        gf_imp = compute_impurity_gf(z, siam, beta, ref)
        sigma = self_energy(gf_imp0, gf_imp)

        # Compute new hybridization
        qp_weight = quasiparticle_weight(z.real, sigma, thresh=vthresh)
        if not np.isfinite(qp_weight):
            raise RuntimeError(f"Quasparticle weight is not finite: {qp_weight}")
        v_new = mix_values(v, np.sqrt(qp_weight * m2), mixing=mixing)

        # Compute errors and update hybridization
        delta_v = np.linalg.norm(v - v_new)
        v = v_new

        # logger.debug("Iteration %d: V=%s, qp-weight=%s, Δv=%.4E", it, v, qp_weight, delta_v)
        # Break if convergence of hybridization
        if v == 0. or delta_v < vtol * mixing:
            converged = True
            break

    # Update final hybridization value and iteration status
    siam.update_hybridization(v)
    logger.info("U:         %.2f", u)
    logger.info("Iteration: %s", it)
    logger.info("Success:   %s", converged)
    logger.info("Params:    v=%.4f, eps=%.4f", siam.v, siam.eps_bath)
    logger.info("Errors:    Δv=%.4E", delta_v)
    logger.info("-"*40)
    return siam


def plot_gf(u=4, eps=0, v=1, ref=False, beta=200):
    num_bath = 1
    eps = eps * np.ones(num_bath)
    v = v * np.ones(num_bath)
    model = SingleImpurityAndersonModel(u=u, eps_bath=eps, v=v, mu=u / 2)

    # gs = compute_groundstate(model)
    # print(gs)
    z = np.linspace(-10, +10, 1000) + 1e-2j
    gf = compute_impurity_gf(z, model, beta, ref=ref)
    gf0 = model.impurity_gf0(z)
    sigma = self_energy(gf0, gf)

    plt.plot(z.real, -gf0.imag)
    plt.plot(z.real, -gf.imag)
    plt.plot(z.real, -sigma.imag)
    # plt.plot(z.real, -impurity_gf_ref(z, model.u, model.v).imag)
    plt.show()


def plot_qp_weight():
    temp = 0.05
    beta = 100
    z = np.linspace(-6, +6, 1000) + 1e-4j
    ref = False
    u_values = np.arange(0, 10, 0.2)
    qp_weights = np.zeros_like(u_values)
    v_vals = np.zeros_like(u_values)
    for i, u in enumerate(u_values):
        siam = twosite_dmft_half_filling(z, u=u, t=1.0, beta=beta, ref=ref)
        v_vals[i] = siam.v[0]

        gfimp_0 = siam.impurity_gf0(z)
        gfimp = compute_impurity_gf(z, siam, beta, ref=ref)
        sigma = self_energy(gfimp_0, gfimp)
        qp_weight = quasiparticle_weight(z.real, sigma)
        qp_weights[i] = qp_weight

    idx = 20
    siam = SingleImpurityAndersonModel(u=u_values[idx], v=v_vals[idx], mu=u_values[idx]/2)
    gf = compute_lattice_greens_function(z, siam, t=1., beta=beta)

    fig, ax = plt.subplots()
    ax.plot(z.real, -gf.imag)
    ax.set_title(f"Lattice GF (U={u_values[idx]:.2f})")

    fig, ax = plt.subplots()
    ax.plot(u_values, qp_weights, color="C1")
    ax.set_title(f"Quasi-particle weight")
    ax.grid()
    plt.show()


def main():
    z = np.linspace(-6, +6, 1000) + 1e-4j
    # twosite_dmft_half_filling(z, u=4.2, beta=100, ref=False)
    # gs = compute_groundstate(model)
    # plot_gf()
    plot_qp_weight()


if __name__ == "__main__":
    main()
