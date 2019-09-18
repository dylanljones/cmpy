# -*- coding: utf-8 -*-
"""
Created on 11 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from sciutils import Plot, use_cycler, Terminal, save_pkl, load_pkl, Path
from cmpy.core import get_omegas, spectral
from cmpy.dmft.two_site import TwoSiteDmft

use_cycler()

WIDTH = 500
IMG = r"_data/figs"
DATA = Path(r"_data/dmft/two_site", init=True)


def calculte_lattice_gf(z, t=1., thresh=1e-4, mixing=0.):
    u_values = np.arange(0, 5.1, 0.5)
    for i, u in enumerate(u_values):
        print(f"{i} Calculating lattice Gf (u={u})")
        solver = TwoSiteDmft(z, u=u, t=t)
        solver.solve_self_consistent(thresh)
        data = [solver.omega, solver.gf_latt]
        save_pkl(Path(DATA, f"gf_latt_{i}.pkl"), data, info=u)


def plot_lattice_gfs():
    folder = Path(DATA)
    plot = Plot(xlabel=r"$\omega$", ylabel=r"$G_{latt}$", width=WIDTH)
    for file in folder.search_files("gf_latt_"):
        data, u = load_pkl(file)
        omega, gf = data
        plot.plot(omega, -gf.imag, label=r"$U=$" + f"{u}")
    plot.grid()
    plot.legend()
    plot.tight()
    # plot.save(IMG, "twosite_half_z.png")
    plot.show()


def plot_lattice_gf(u=1.0):
    folder = Path(DATA)

    for file in folder.search_files("gf_latt_"):
        data, u_val = load_pkl(file)
        if u_val == u:
            data, u = load_pkl(file)
            omega, gf = data
            plot = Plot(xlabel=r"$\omega$", ylabel=r"$G_{latt}$", width=WIDTH)
            plot.plot(omega, -gf.imag, label=r"$U=$" + f"{u}")
            plot.grid()
            plot.legend()
            plot.tight()
            # plot.save(IMG, "twosite_half_z.png")
            plot.show()



def calculate_quasiparticle_weight(z, t=1., thresh=1e-4, mixing=0.):
    u_values = np.linspace(0.01, 8, 50)
    betas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    n = len(u_values)
    idx = 0
    cout = Terminal()
    for beta in betas:
        header = f"Beta: {str(beta):<6} "
        z_values = np.zeros_like(u_values)
        cout.write(header)
        for i in range(n):
            cout.updateln(header + f"Calculating: {100*i/n:.1f}%")
            solver = TwoSiteDmft(z, u=u_values[i], t=t, beta=beta)
            solver.solve_self_consistent(thresh, mixing=mixing, nmax=1000)
            z_values[i] = solver.quasiparticle_weight
        cout.updateln(header + "done!")
        cout.writeln("")
        save_pkl(Path(DATA, f"qpweight_{idx}.pkl"), u_values, z_values, info=beta)
        idx += 1


def plot_quasiparticle_weights():
    folder = Path(DATA)
    plot = Plot(xlabel=r"$U$", ylabel=r"$z(u)$", width=WIDTH)
    for file in folder.search_files("qpweight_"):
        u, z, beta = load_pkl(file)
        plot.plot(u, z, label=r"$\beta=$" + f"{beta}")
    plot.set_limits(xlim=0, ylim=[0, 1.05])
    plot.grid()
    plot.legend()
    plot.tight()
    # plot.save(IMG, "twosite_half_z.png")
    plot.show()


def main():
    u = 3
    t = 2
    thresh = 1e-5
    omegas = np.linspace(-8, 8, 10000)
    eta = 0.01j
    z = omegas + eta
    # ----------------------------------------------

    # calculte_lattice_gf(z, t, thresh, mixing=0.)
    # calculate_quasiparticle_weight(omegas + eta, t, thresh, mixing=0.)
    plot_lattice_gf(5)
    plot_quasiparticle_weights()

if __name__ == "__main__":
    main()
