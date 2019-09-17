# -*- coding: utf-8 -*-
"""
Created on 11 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from sciutils import Plot, Colors
from cmpy.core import get_omegas
from cmpy.dmft.two_site import TwoSiteDmft


def plot_solver(omegas, u):
    solver = TwoSiteDmft(omegas, u)
    solver.solve()

    print(solver.quasiparticle_weight())
    plot = Plot(width=600, height=300, ylabel=r"$G_{latt}$")
    plot.plot(omegas, solver.dos_latt(), color="k")
    plot.add_yax()
    plot.plot(omegas, -solver.sigma.imag, Colors.bblue, label=r"$\Sigma$", ls="--")
    plot.set_labels(ylabel=r"$\Sigma$")
    plot.yaxis.label.set_color(Colors.bblue)
    plot.show()


def get_lines(omegas, u_values):
    lines = list()
    for u in u_values:
        print(u)
        solver = TwoSiteDmft(omegas, u)
        solver.solve()
        lines.append(solver.dos_latt())
    return lines


def run_dmft(solver, v0=1., thresh=1e-5):
    v = v0
    diff = 10
    while diff > thresh:
        print(f"v={float(v):.4} (delta={float(diff):.4})")
        solver.siam.update_hopping(v)
        solver.solve()

        z = solver.quasiparticle_weight()
        m2 = solver.m2_weight()
        new_v = np.sqrt(z * m2)
        new_v = (new_v + v) / 2  # Mixing

        diff = abs(v - new_v)
        v = new_v


def main():
    u = 2
    omegas, eta = get_omegas(10, 10000, 0.5)
    solver = TwoSiteDmft(omegas, u)
    # ----------------------------------------------
    run_dmft(solver)


if __name__ == "__main__":
    main()
