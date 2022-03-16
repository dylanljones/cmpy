# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2022, Dylan Jones

import numpy as np
from cmpy.models.anderson import SingleImpurityAndersonModel
from .utils import (
    IterationStats,
    self_energy,
    quasiparticle_weight,
    mix_values,
    bethe_gf_omega,
)


# ========================== REFERENCES =======================================

# Reference functions taken from E. Lange:
# 'Renormalized vs. unrenormalized perturbation-theoretical
# approaches to the Mott transition'


def impurity_params_ref(u: float, v: float):
    r"""Poles and residues of the two-site SIAM  Greens function at half filling and T=0.

    Parameters
    ----------
    u : float
        The on-site interaction strength.
    v : float
        The hopping energy between the impurity site and the bath site.

    Returns
    -------
    alpha_1 : float
        The first residuum of the Greens function.
    alpha_2 : float
        The second residuum of the Greens function.
    omega_1 : float
        The first residuum of the Greens function.
    omega_2 : float
        The second residuum of the Greens function.
    """
    sqrt16 = np.sqrt(u**2 + 16 * v**2)
    sqrt64 = np.sqrt(u**2 + 64 * v**2)
    a1 = (
        1
        / 4
        * (
            1
            - (u**2 - 32 * v**2)
            / np.sqrt((u**2 + 64 * v**2) * (u**2 + 16 * v**2))
        )
    )
    a2 = 1 / 2 - a1
    e1 = 1 / 4 * (sqrt64 - sqrt16)
    e2 = 1 / 4 * (sqrt64 + sqrt16)
    return float(a1), float(a2), float(e1), float(e2)


def impurity_gf_ref(z: np.ndarray, u: float, v: float) -> np.ndarray:
    r"""Impurity Greens function of the two-site SIAM at half filling and zero temperature.

    Parameters
    ----------
    z : (N) np.ndarray
        The broadened complex frequency .math:`\omega + i \eta`.
    u : float
        The on-site interaction strength.
    v : float
        The hopping energy between the impurity site and the bath site.

    Returns
    -------
    gf_imp : (N) np.ndarray
        The interacting impurity Greens function.
    """
    a1, a2, e1, e2 = impurity_params_ref(u, v)
    return (a1 / (z - e1) + a1 / (z + e1)) + (a2 / (z - e2) + a2 / (z + e2))


# =========================================================================


def impurity_gf0(z, siam):
    return siam.impurity_gf0(z)


def compute_impurity_gf(z, siam, ref=False):
    if ref:
        return impurity_gf_ref(z, siam.u, siam.v)
    return siam.impurity_gf(z)


def compute_self_energy(z, siam, ref=False):
    gf_imp0 = impurity_gf0(z, siam)
    gf_imp = compute_impurity_gf(z, siam, ref=ref)
    return self_energy(gf_imp0, gf_imp)


def twosite_dmft_half_filling(
    z,
    u,
    t=1.0,
    beta=np.inf,
    mixing=1.0,
    vtol=1e-6,
    max_iter=1000,
    vthresh=1e-10,
    verbose=True,
    ref=True,
):

    # Map the lattice model to the single impurity Anderson model at half filling
    u = u
    v = [t]

    siam = SingleImpurityAndersonModel(u, v=v, mu=u / 2, temp=1 / beta)

    # Initial hybridization value must be different
    # from the current one or the error will be zero.
    v = siam.v[0] + 0.1
    # prepare m2 weight
    m2 = t**2

    it = 0
    stats = IterationStats("Δv")
    while True:
        # Update parameters of SIAM
        siam.update_hybridization(v)

        # Solve impurity problem to obtain the self energy
        gf_imp0 = impurity_gf0(z, siam)
        gf_imp = compute_impurity_gf(z, siam, ref=ref)
        sigma = self_energy(gf_imp0, gf_imp)

        # Compute new hybridization
        qp_weight = quasiparticle_weight(z.real, sigma, thresh=vthresh)
        v_new = mix_values(v, np.sqrt(qp_weight * m2), mixing=mixing)

        # Compute errors and update stats
        delta_v = np.linalg.norm(v - v_new)
        stats.append(delta_v)
        v = v_new

        # Break if convergence of hybridization or the
        # maximum iteration number is reached.
        if v == 0:
            stats.set_parameter_converged("Hybridization", v)
            break
        if delta_v < vtol:
            stats.set_parameter_converged("Hybridization", v)
            break
        elif qp_weight == 0:
            stats.set_parameter_converged("Quasiparticle weight", qp_weight)
            break
        elif it >= max_iter:
            stats.set_maxiter_status(max_iter)
            break
        it += 1

    # Update final hybridization value and iteration status
    siam.update_hybridization(v)

    if verbose:
        print("-" * 50)
        print(f"U:          {u:.2f}")
        print(stats)

    return siam


def compute_lattice_greens_function(z, siam, t, ref=False):
    mu = siam.mu
    sigma = compute_self_energy(z, siam, ref)
    return bethe_gf_omega(z + mu - sigma, t)
