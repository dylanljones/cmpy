# coding: utf-8
#
# This code is part of cmpy.
# 
# Copyright (c) 2021, Dylan Jones

import logging
import numpy as np
import matplotlib.pyplot as plt
from cmpy import UP
from cmpy.models import SingleImpurityAndersonModel
from cmpy import exactdiag as ed

logger = logging.getLogger("cmpy")
logger.setLevel(logging.DEBUG)


def greens_function_lehmann(model, z, beta, pos=0, sigma=UP):
    data = ed.GreensFunctionMeasurement(z, beta, pos, sigma)
    eig_cache = dict()
    occ = 0.
    occ_double = 0.
    for n_up, n_dn in model.iter_fillings():
        sector = model.get_sector(n_up, n_dn)
        eigvals, eigvecs = ed.solve_sector(model, sector, cache=eig_cache)

        occ += ed.occupation(sector, eigvals, eigvecs, beta, pos=pos, sigma=sigma)
        occ_double += ed.double_occupation(sector, eigvals, eigvecs, beta, pos=pos)
        # Check if upper particle sector exists
        sector_p1 = model.basis.upper_sector(n_up, n_dn, sigma)
        if sector_p1 is not None:
            eigvals_p1, eigvecs_p1 = ed.solve_sector(model, sector_p1, cache=eig_cache)
            data.accumulate(sector, sector_p1, eigvals, eigvecs, eigvals_p1, eigvecs_p1)
        else:
            eig_cache.clear()

    print(f"Occ-up {occ / data.part:.6f}")
    print(f"Occ-up {data.occ:.6f}")
    print(f"double occ-up {occ_double / data.part:.6f}")
    print(f"double occ-up {data.occ_double:.6f}")
    return data.gf


def main():
    beta = 100
    num_bath = 1
    u = 2
    eps_imp = 0
    eps_bath = 0 * np.random.uniform(size=num_bath)
    v = np.ones(num_bath)

    siam = SingleImpurityAndersonModel(u, eps_imp, eps_bath, v, mu=None)

    z = np.linspace(-7, +7, 1000) + 0.01j
    data = ed.greens_function_lehmann(siam, z, beta)
    gf = data.gf

    fig, ax = plt.subplots()
    ax.plot(z.real, -gf.imag)
    ax.set_xlim(min(z.real), max(z.real))
    ax.grid()
    plt.show()


if __name__ == "__main__":
    main()
