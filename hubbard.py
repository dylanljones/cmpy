# -*- coding: utf-8 -*-
"""
Created on 29 Mar 2019
author: Dylan

project: cmpy2
version: 1.0
"""
import numpy as np
from sciutils import Plot, eta
from cmpy.hubbard import HubbardModel, State, Siam


def main():
    eps = np.zeros(5)
    v = np.random.uniform(0, 2, size=10)
    model = Siam(u=2, mu=0)
    # model = HubbardModel(shape=(2, 1), mu=0)

    omegas = np.linspace(-5, 15, 2000) + 0.02j

    plot = Plot(xlabel=r"$\omega$", ylabel=r"$A(\omega)$")
    plot.plot(omegas.real, model.spectral(omegas))
    plot.set_limits(0)
    # plot.set_ticks([-10, -5, 0, 5, 10])
    plot.show()


if __name__ == "__main__":
    main()
