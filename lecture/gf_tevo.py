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

logger = logging.getLogger("cmpy")
logger.setLevel(logging.DEBUG)


def main():
    num_bath = 2
    u = 5
    eps_imp = 0
    eps_bath = 1 * np.random.uniform(size=num_bath)
    v = np.ones(num_bath)
    siam = SingleImpurityAndersonModel(u, eps_imp, eps_bath, v, mu=None)

    start, stop, dt = 0, 150, 0.05
    times, gf_t = ed.greens_function_tevo(siam, start, stop, num=int(stop / dt) + 1)

    fig, ax = plt.subplots()
    ax.plot(times, -gf_t.imag)
    ax.set_xlim(times[0], times[-1])
    # ax.set_ylim(-1.1, +1.1)
    ax.grid()

    w = np.linspace(-7, +7, 5000)
    z, gf_w = ed.fourier_t2z(times, gf_t, w, delta=1e-4)

    fig, ax = plt.subplots()
    ax.plot(z.real, -gf_w.imag)
    ax.set_xlim(min(z.real), max(z.real))
    ax.grid()

    plt.show()


if __name__ == "__main__":
    main()
