# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2022, Dylan Jones

import logging
import matplotlib.pyplot as plt
import numpy as np
import lattpy as lp
from cmpy import UP, CreationOperator, AnnihilationOperator
from cmpy.exactdiag import solve_sector
from cmpy.models import HubbardModel

logger = logging.getLogger("cmpy")
logger.setLevel(logging.DEBUG)


def _acc_lehman(gf, z, c_evec, cdag_evec, evals, evecs, evals_p1, evecs_p1, beta, emin):
    overlap = np.abs(evecs_p1.T.conj() @ cdag_evec) ** 2  # |<m|c_i^†|n>|^2
    overlap = np.abs(evecs.T.conj() @ c_evec) ** 2  # |<n|c_i|m>|^2
    overlap2 = evecs_p1.T.conj() @ cdag_evec  # <m|c_j^†|n>
    overlap1 = evecs.T.conj() @ c_evec  # <n|c_i|m>
    # overlap = np.dot(overlap1, overlap2)             # <n|c_i|m><m|c_j^†|n>
    # print(overlap.shape)
    if np.isfinite(beta):
        exp_evals = np.exp(-beta * (evals - emin))
        exp_evals_p1 = np.exp(-beta * (evals_p1 - emin))
    else:
        exp_evals = np.ones_like(evals)
        exp_evals_p1 = np.ones_like(evals_p1)

    num_m = len(evals_p1)
    num_n = len(evals)
    for m in range(num_m):
        eig_m = evals_p1[m]
        z_m = z - eig_m
        for n in range(num_n):
            eig_n = evals[n]
            weights = exp_evals[n] + exp_evals_p1[m]
            gf += overlap[n, m] * weights / (z_m + eig_n)
            # gf += overlap1[n, m] * overlap2[m, n] * weights / (z_m + eig_n)


def accumulate_gf(gf, z, cop, cdag, evals, evecs, evals_p1, evecs_p1, beta, emin=0.0):
    cdag_evec = cdag.matmat(evecs)  # c^† |n>
    c_evec = cop.matmat(evecs_p1)  # c |m>
    return _acc_lehman(
        gf, z, c_evec, cdag_evec, evals, evecs, evals_p1, evecs_p1, beta, emin
    )


def greens_lehmann(model, z, beta, i, j, sigma=UP, occ=True, eig_cache=None):
    basis = model.basis
    eig_cache = eig_cache if eig_cache is not None else dict()
    part = 0.0

    logger.info("Accumulating Lehmann sum (i=%s, j=%s, sigma=%s)", i, j, sigma)
    logger.debug("Sites: %s (%s states)", basis.num_sites, basis.size)

    e0 = np.infty
    gf = np.zeros_like(z)
    fillings = list(basis.iter_fillings())
    num = len(fillings)
    w = len(str(num))
    for i, (n_up, n_dn) in enumerate(fillings):
        logger.info("[%s/%s] Sector %s, %s", f"{i+1:>{w}}", num, n_up, n_dn)

        sector = model.get_sector(n_up, n_dn)
        sector_p1 = basis.upper_sector(n_up, n_dn, sigma)
        if sector_p1 is not None:
            # Solve sectors
            evals, evecs = solve_sector(model, sector, cache=eig_cache)
            evals_p1, evecs_p1 = solve_sector(model, sector_p1, cache=eig_cache)

            # Update Ground state energy
            min_energy = min(evals)
            factor = 1.0
            if min_energy < e0:
                factor = np.exp(-beta * (e0 - min_energy))
                e0 = min_energy

            logger.debug("Accumulating")

            # Accumulate partition function
            part *= factor
            part += np.sum(np.exp(-beta * (evals - e0)))

            # Accumulate Green's function
            if factor != 1.0:
                gf *= factor
            cop = AnnihilationOperator(sector_p1, sector, pos=i, sigma=sigma)
            cdag = CreationOperator(sector, sector_p1, pos=j, sigma=sigma)
            accumulate_gf(gf, z, cop, cdag, evals, evecs, evals_p1, evecs_p1, beta, e0)
        else:
            logger.debug("No upper sector, skipping")
    return gf


def main():
    latt = lp.finite_hypercubic(2)
    u = 2.0

    model = HubbardModel(latt, inter=u, mu=u / 2)

    z = np.linspace(-10, +10, 1000) + 1e-2j

    gf = greens_lehmann(model, z, beta=10.0, i=0, j=0)

    plt.plot(z.real, -gf.imag)
    plt.show()


if __name__ == "__main__":
    main()
