# -*- coding: utf-8 -*-
"""
Created on 26 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from cmpy import square_lattice, eta, plot_transmission
from cmpy.core import *
from cmpy.tightbinding import square_device, TbDevice, s_basis, sp3_basis
from scipy.sparse import coo_matrix

POINTS = [0, 0], [np.pi, 0], [np.pi, np.pi]
PNAMES = r"$\Gamma$", r"$X$", r"$M$"


def ham_data(tb):
    for i in prange(tb.lattice.n, header="Building hamiltonian"):
        # Site energies
        yield i, i, tb.get_energy(i)
        # Hopping energies
        neighbours = tb.lattice.neighbours[i]
        for dist in range(len(neighbours)):
            t = tb.hoppings[dist]
            for j in neighbours[dist]:
                if j > i:
                    yield i, j, t
                    yield j, i, np.conj(t).T


def coo_hamiltonian(device):
    rows, cols, data = list(), list(), list()
    for i, j, t in ham_data(device):
        rows.append(i)
        cols.append(j)
        data.append(t)
    n = device.lattice.n
    ham = coo_matrix((data, (rows, cols)), shape=(n, n))

    return ham


def main():
    print("Building lattice...")
    device = TbDevice.square(shape=(100, 1))
    print("done")
    coo_ham = coo_hamiltonian(device)
    print(coo_ham.row)
    print(coo_ham.col)
    print(coo_ham.data)

    ham = Hamiltonian(coo_ham.toarray())
    ham.show()


if __name__ == "__main__":
    main()
