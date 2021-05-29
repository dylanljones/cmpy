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
    tmax, dt = 300, 0.1
    start, stop, num = 0, tmax, int(tmax / dt)
    num_bath = 1
    u = 2
    eps_imp = 0
    eps_bath = 0 * np.random.uniform(size=num_bath)
    v = np.ones(num_bath)

    siam = SingleImpurityAndersonModel(u, eps_imp, eps_bath, v, mu=None)

    times, gf_t = ed.greens_function_tevo(siam, start, stop, num)

    fig, ax = plt.subplots()
    ax.plot(times, -gf_t.imag)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-1.1, +1.1)
    ax.grid()

    w = np.linspace(-7, +7, 5000)
    z, gf_w = ed.fourier_t2z(times, gf_t, w, delta=1e-2)

    fig, ax = plt.subplots()
    ax.plot(z.real, -gf_w.imag)
    ax.set_xlim(min(z.real), max(z.real))
    ax.grid()

    plt.show()


if __name__ == "__main__":
    main()
