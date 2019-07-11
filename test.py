# -*- coding: utf-8 -*-
"""
Created on 26 Mar 2019
author: Dylan

project: cmpy2
version: 1.0
"""
from cmpy2 import *
from cmpy2.tightbinding import *


def dos(ham, energies):
    n = len(energies)
    dos = np.zeros(n, "float")
    for i in range(n):
        gf = greens.greens(ham, energies[i] + eta)
        dos[i] = np.trace(gf.imag)
    return dos


def main():

    model = TbDevice.square((1000, 1), eps=0.5, t=1)
    omegas = np.linspace(-3, 3, 100)

    ham = model.hamiltonian()
    # ham.show()

    plot = Plot()
    # plot.plot(omegas, model.dos_device(omegas + eta))
    plot.plot(omegas, dos(ham, omegas))
    plot.show()

if __name__ == "__main__":
    main()
