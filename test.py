# -*- coding: utf-8 -*-
"""
Created on 26 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
from cmpy import *
from cmpy.tightbinding import *
from cmpy.hubbard import *
from scipy.integrate import quad


# =========================================================================


def func(x, omega, mu, sigma, t=1):
    return 1/(2*np.pi*t) * np.sqrt(2*t*t - x*x) / (omega + mu - x - sigma)


def plot_arg():
    x = np.linspace(-2, 2, 100)
    y = func(x, 0, 1, 1, 1)
    plot = Plot()
    plot.plot(x, y)
    plot.show()


def main():
    model = HubbardModel()
    ham = model.hamiltonian()
    ham.show()

    omegas = np.linspace(-20, 20, 100) + eta
    spectral = -np.sum(ham.greens(omegas, only_diag=True), axis=1)
    plot = Plot()
    plot.plot(omegas.real, spectral)
    plot.show()
    print(model)






if __name__ == "__main__":
    main()
