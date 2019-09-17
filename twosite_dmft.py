# -*- coding: utf-8 -*-
"""
Created on 11 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from sciutils import Plot, use_cycler
from cmpy.core import get_omegas, spectral
from cmpy.dmft.two_site import TwoSiteDmft

use_cycler()

WIDTH = 500
IMG = r"_data/figs"


def plot_lattice_gf(solver):
    solver.solve()
    plot = Plot(width=600, height=300, xlabel=r"$\omega$", ylabel=r"$G_{latt}$")
    plot.plot(solver.omega, spectral(solver.gf_latt))
    plot.show()


def run_dmft(z, u, t, thresh=1e-5):
    solver = TwoSiteDmft(z, u=u, t=t)
    solver.solve_self_consistent(thresh)
    return solver


def plot_lattice_gf(i, z, u, t=1, thresh=1e-5, show=False):
    solver = run_dmft(z, u, t, thresh)

    plot = Plot(xlabel=r"$\omega$", ylabel=r"$G_{latt}$", width=WIDTH)
    plot.set_title(f"Lattice Green's function ($u={u}, t={t}$)")
    plot.plot(solver.omega, spectral(solver.gf_latt))
    plot.tight()
    plot.save(IMG, f"gf_lattice_{i}.png")
    if show:
        plot.show()


def plot_quasiparticle_weights(omegas, u_values, t=1., thresh=1e-4, mixing=0., show=False):
    z_values = np.zeros_like(u_values)
    for i, u in enumerate(u_values):
        solver = TwoSiteDmft(omegas, u=u_values[i], t=t)
        solver.solve_self_consistent(thresh, mixing=mixing, header=f"{i: <4}")
        z_values[i] = solver.quasiparticle_weight

    plot = Plot(xlabel=r"$\omega$", ylabel=r"$z$", width=WIDTH)
    plot.set_title(f"Quasiparticle weight ($t={t}$)")
    plot.plot(u_values, z_values)
    plot.tight()
    plot.save(IMG, "twosite_half_z.png")
    if show:
        plot.show()


def main():
    u = 4
    t = 1
    thresh = 1e-5
    omegas, eta = get_omegas(8, 10000, 0.25)
    z = omegas + eta
    # ----------------------------------------------
    # u_values = np.arange(0.25, 5, 0.25)
    # for i, u in enumerate(u_values):
    #     plot_lattice_gf(i, z, u, t, thresh)
    u_values = np.linspace(0.01, 4, 200)
    plot_quasiparticle_weights(omegas + eta, u_values, t, thresh, mixing=0.)


if __name__ == "__main__":
    main()
