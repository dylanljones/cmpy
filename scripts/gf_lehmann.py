# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2022, Dylan Jones

import logging
import numpy as np
import matplotlib.pyplot as plt
from cmpy.models import SingleImpurityAndersonModel
from cmpy import exactdiag as ed

logger = logging.getLogger("cmpy")
logger.setLevel(logging.DEBUG)


def main():
    beta = 100
    num_bath = 4
    u = 2
    eps_imp = 0
    eps_bath = 1 * np.random.uniform(size=num_bath)
    v = np.ones(num_bath)

    siam = SingleImpurityAndersonModel(u, eps_imp, eps_bath, v, mu=None)

    z = np.linspace(-10, +10, 1000) + 0.01j
    data = ed.gf_lehmann(siam, z, beta)
    gf = data.gf

    fig, ax = plt.subplots()
    ax.plot(z.real, -gf.imag)
    ax.set_xlim(min(z.real), max(z.real))
    ax.grid()
    plt.show()


if __name__ == "__main__":
    main()
