# -*- coding: utf-8 -*-
"""
Created on 26 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
from cmpy import *
from cmpy.tightbinding import *

POINTS = [0, 0], [np.pi, 0], [np.pi, np.pi]
NAMES = r"$\Gamma$", r"$X$", r"$M$"

# =========================================================================


def center_ticks(n, limit=20):
    if n < limit:
        ticks = np.arange(n) + 0.5
        labels = [str(x) for x in np.arange(n) + 1]
    else:
        ticks = 0.5, n-0.5
        labels = [1, str(n)]
    return ticks, labels


def plot_local_occupation(model, occ):
    n_x, n_y = model.lattice.shape
    xx, yy = np.meshgrid(range(n_x+1), range(n_y+1))
    occ = np.reshape(occ, (n_x, n_y))

    plot = Plot()
    plot.set_equal_aspect()
    im = plot.colormesh(xx, yy, occ.T, cmap="Greys")
    plot.fig.colorbar(im, ax=plot.ax)

    ticks, labels = center_ticks(n_y)
    plot.set_ticks(yticks=ticks)
    plot.set_ticklabels(yticks=labels)

    ticks, labels = center_ticks(n_x)
    plot.set_ticks(xticks=ticks)
    plot.set_ticklabels(xticks=labels)

    plot.show()


def occupation_probability(model, idx=0):
    ham = model.hamiltonian(model.w_eps)
    eigvals, eigvecs = ham.eig()
    i_min = np.argmin(eigvals)
    psi = eigvecs[i_min]
    occ = np.abs(psi * psi)
    print(occ)
    print(sum(occ))
    plot_local_occupation(model, occ)


def ipr(psi):
    occ = np.abs(psi * psi)
    return np.sum(occ**2)


def ipr_curve(model, n_avrg=100):
    n = 100
    w_values = np.linspace(0, 10, n)
    ipr_vals = np.zeros(n)
    for i in prange(n):
        tmp_ipr = np.zeros(n_avrg)
        for j in range(n_avrg):
            ham = model.hamiltonian(w_values[i])
            eigvals, eigstates = ham.eig()
            i_min = np.argmin(eigvals)
            tmp_ipr[j] = ipr(eigstates[i_min])

        ipr_vals[i] = np.mean(tmp_ipr)
    return w_values, ipr_vals


def plot_ipr(model):
    plot = Plot()
    w, ipr_vals = ipr_curve(model)
    plot.plot(w, ipr_vals)
    plot.show()


def main():

    model = TbDevice.square((20, 5))
    # model.show()
    model.set_disorder(0)

    # dos_probability(model)
    # occupation_probability(model)
    print(1/model.lattice.n)
    plot_ipr(model)







if __name__ == "__main__":
    main()
