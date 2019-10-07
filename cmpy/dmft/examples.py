# -*- coding: utf-8 -*-
"""
Created on 15 Aug 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from scitools import Plot
from cmpy import spectral
from cmpy.dmft import solvers, quasiparticle_weight


# =========================================================================
# Iterative Perturbation Theory (ITP)
# =========================================================================


def plot_linemesh(lines, size=5., color="k", *args, **kwargs):
    plot = Plot()
    dy = size/len(lines)
    for i, data in enumerate(lines):
        x, y = data
        shift = i * dy
        plot.draw_lines(y=shift, color="0.5", lw=0.5)
        plot.plot(x, y + shift, *args, color=color, **kwargs)
    return plot


def plot_bandmesh(temp=0.1, t=1.):
    omegas = np.linspace(-4, 4, 1000)
    u_values = np.arange(1, 5, 0.1)
    lines = list()
    for i, u in enumerate(reversed(u_values)):
        gf, sigma = solvers.ipt.real_loop(omegas, 1/temp, u, t)
        lines.append((omegas, spectral(gf)))

    plot = plot_linemesh(lines, size=2.5)
    plot.set_limits(0)
    plot.set_labels(r"$\omega$", r"$A(\omega, U)$")
    plot.show()


def plot_quasiparticle_weight(temp, t=0.5, ulow=0, uhigh=5, n=1000):
    beta = 1/temp
    u_values = np.linspace(ulow, uhigh, n)
    qp_weight = np.zeros(n)
    for i in range(n):
        omega, gf, sigma = solvers.ipt.imag_loop(beta, u_values[i], t, n=100)
        qp_weight[i] = quasiparticle_weight(sigma, beta)

    plot = Plot()
    plot.set_labels("U/D", "z")
    plot.plot(u_values, qp_weight, label=r"$\beta$" + f"$={beta:.2f}$")
    plot.legend()
    plot.show()


def main():
    t = 0.5
    temp = 0.01
    # ITP
    plot_bandmesh(temp, t)
    # plot_quasiparticle_weight(temp, t)


if __name__ == "__main__":
    main()
