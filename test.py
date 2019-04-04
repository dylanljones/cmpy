# -*- coding: utf-8 -*-
"""
Created on 29 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
import os
import time
import numpy as np
from cmpy import SparseHamiltonian
from scipy.sparse import coo_matrix



def main():
    n = 5
    n_orbs = 2
    m = SparseHamiltonian(n, n_orbs)
    m.set_all_energies(np.eye(n_orbs)*2)
    for i in range(n-1):
        m.set_hopping(i, i+1, np.eye(n_orbs))
    print(m)





if __name__ == "__main__":
    main()
