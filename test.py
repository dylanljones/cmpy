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
    model = TightBinding(np.eye(2))
    model.add_atom()
    model.add_atom("B", pos=np.array([0.5, 0]))
    model.set_hopping(1)
    print(model.cell_hamiltonian())
    model.build((5, 1))
    model.lattice.show()
    model.plot_bands(POINTS, NAMES)
    #ham = model.hamiltonian()
    # print(ham)
    # model.hamiltonian(1).show()


if __name__ == "__main__":
    main()
