# -*- coding: utf-8 -*-
"""
Created on 11 Nov 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from cmpy import FBasis
from cmpy import Lattice, FState, FBasis
from cmpy.core.math import fermi_dist
from scitools import Plot


def main():
    basis = FBasis.free(5)
    for i in range(1, basis.n_sites):
        print(basis[0].check_hopping(basis[i], 1))
        print(basis[0].hopping_indices(basis[i]))
        print()


if __name__ == "__main__":
    main()
