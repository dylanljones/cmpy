# -*- coding: utf-8 -*-
"""
Created on 26 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
from itertools import product
from sciutils import eta, Plot
from cmpy import *
from cmpy.tightbinding import *
from cmpy.hubbard import *

# =========================================================================


def main():
    model = TbDevice.square((100, 100))
    omegas = np.linspace(-3, 3, 1000) + 0.01j

    ham = model.hamiltonian()
    gf = gf_lehmann(ham, omegas)
    dos = -1/np.pi * np.sum(gf.imag, axis=1)

    Plot.quickplot(omegas.real, dos)


if __name__ == "__main__":
    main()
