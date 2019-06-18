# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2018
@author: Dylan Jones

project: ipr$
version: 1.0
"""
import numpy as np
from cmpy import Plot, prange
from cmpy.tightbinding import TbDevice


def ipr(psi):
    occ = np.abs(psi * psi)
    return np.sum(occ**2)


def mean_ipr(model, w_eps, e_idx, n_avrg=200):
    tmp = np.zeros(n_avrg)
    for i in range(n_avrg):
        ham = model.hamiltonian(w_eps)
        _, eigvecs = ham.eig()
        psi = eigvecs[0]
        tmp[i] = np.sum(np.power(np.abs(psi), 4))
    return np.mean(tmp)


def ipr_curve(model, n_avrg=100):
    n = 50
    w_values = np.linspace(0, 10, n)
    ipr_vals = np.zeros(n)
    for i in prange(n):
        ipr_vals[i] = mean_ipr(model, w_values[i], 0)
    return w_values, ipr_vals


def plot_ipr(model):
    plot = Plot()
    w, ipr_vals = ipr_curve(model)
    plot.plot(w, ipr_vals)
    plot.show()


def main():
    model = TbDevice.square((20, 5))
    plot_ipr(model)


if __name__ == "__main__":
    main()
