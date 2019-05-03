# -*- coding: utf-8 -*-
"""
Created on 26 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
import re
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cmpy import *
from cmpy.tightbinding import *

POINTS = [0, 0], [np.pi, 0], [np.pi, np.pi]
NAMES = r"$\Gamma$", r"$X$", r"$M$"


def main():
    model = TbDevice.square((2, 1))
    ham = model.hamiltonian()

    n = 1000
    omegas = np.linspace(-5, 5, n)
    y = np.zeros(n)
    for i in range(n):
        gf = greens.greens(ham, omegas[i] + 0.01j)
        dos = -1/np.pi * np.trace(gf.imag)
        y[i] = dos

    plot = Plot()
    plot.set_figsize(width_pt=455.24411)

    plot.plot(omegas, y)
    plot.save("Test.eps")
    # plot.show()


if __name__ == "__main__":
    main()
