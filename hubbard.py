# -*- coding: utf-8 -*-
"""
Created on 29 Mar 2019
author: Dylan

project: cmpy2
version: 1.0
"""
import numpy as np
from sciutils import Plot, eta
from cmpy.hubbard import HubbardModel, State


def main():
    model = HubbardModel()
    model.set_filling(n=2)

    omegas = np.linspace(-7, 7, 2000) + 0.02j
    plot = Plot(xlabel=r"$\omega$", ylabel=r"$A(\omega)$")
    plot.plot(omegas.real, model.spectral(omegas))

    plot.set_limits(0)
    plot.set_ticks([-5, -2.5, 0, 2.5, 5])
    plot.show()


if __name__ == "__main__":
    main()
