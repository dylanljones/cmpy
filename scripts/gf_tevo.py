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
    num_bath = 3
    u = 4
    eps_imp = 0
    eps_bath = 0 * np.random.uniform(size=num_bath)
    siam = SingleImpurityAndersonModel(u, eps_imp, eps_bath, v=1.0, mu=None)
    # hamop = siam.hamilton_operator()
    # hamop.show()

    start, stop, dt = 0, 150, 0.1
    num = int(stop / dt) + 1
    gs = ed.compute_groundstate(siam)
    times, gf_greater = ed.greens_greater(siam, gs, start, stop, num)
    times, gf_lesser = ed.greens_lesser(siam, gs, start, stop, num)
    gf_t = gf_greater - gf_lesser

    fig, ax = plt.subplots()
    ax.plot(times, -gf_greater.imag, lw=0.5, label="GF$^>$")
    ax.plot(times, -gf_lesser.imag, lw=0.5, label="GF$^<$")
    ax.plot(times, -gf_t.imag, label="GF", color="k", lw=1.0)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(-1.1, +1.1)
    ax.legend()
    ax.grid()

    w = np.linspace(-10, +10, 5000)
    z, gf_w = ed.fourier_t2z(times, gf_t, w, delta=1e-4)

    fig, ax = plt.subplots()
    ax.plot(z.real, -gf_w.imag)
    ax.set_xlim(min(z.real), max(z.real))
    ax.grid()

    plt.show()


if __name__ == "__main__":
    main()
