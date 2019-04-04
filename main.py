# -*- coding: utf-8 -*-
"""
Created on 26 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
import time
import numpy as np
import psutil
import matplotlib.pyplot as plt
from cmpy import square_lattice, eta, plot_transmission
from cmpy.core import *
from cmpy.tightbinding import TbDevice, s_basis, sp3_basis
from scipy.sparse import coo_matrix

POINTS = [0, 0], [np.pi, 0], [np.pi, np.pi]
PNAMES = r"$\Gamma$", r"$X$", r"$M$"


def format_num(num, unit="b", div=1024):
    for scale in ['','k','M','G','T','P','E','Z']:
        if abs(num) < div:
            return f"{num:.1f} {scale}{unit}"
        num /= div


def print_mem():
    vmem = psutil.virtual_memory()
    used = format_num(vmem.used)
    free = format_num(vmem.free)
    total = format_num(vmem.total)
    print(f"Free: {free}, Used: {used}, Total: {total}")


def sparse_ham(tb):
    ham = SparseHamiltonian(tb.lattice.n, tb.n_orbs, "complex")
    for i in prange(tb.lattice.n, header="Building hamiltonian"):
        # Site energies
        ham.set_energy(i, tb.get_energy(i))

        # Hopping energies
        neighbours = tb.lattice.neighbours[i]
        for dist in range(len(neighbours)):
            t = tb.hoppings[dist]
            for j in neighbours[dist]:
                if j > i:
                    ham.set_hopping(i, j, t)
    return ham

def rgf(ham, omega, chunksize=None):
    """ Recursive green's function

    Calculate green's function using the recursive green's function formalism.

    Parameters
    ----------
    ham: Hamiltonian
        hamiltonian of model, must allready be blocked
    omega: float
        energy-value to calculate the Green's functions
    chunksize: int, optional
        chunk size to use in recursion.
        If None, use full pre defined blocks of hamiltonian

    Returns
    -------
    gf_1n: array_like
        lower left block of greens function
    """
    if chunksize is None and not ham.is_blocked:
        raise ValueError("Block sizes of hamiltonian not set up and no chunksize specified!")

    if chunksize is not None:
        ham.config_uniform_blocks(chunksize)

    n_blocks = ham.block_shape[0]
    g_nn = greens(ham.get_block(0, 0), omega)
    g_1n = g_nn
    for i in range(1, n_blocks):
        h = ham.get_block(i, i) + ham.get_block(i, i-1) @ g_nn @ ham.get_block(i-1, i)
        g_nn = greens(h, omega)
        g_1n = g_1n @ ham.get_block(i-1, i) @ g_nn
    return g_1n


def main():
    print("Building lattice...")
    b = sp3_basis()
    dev = TbDevice.square((10, 4), b.eps, b.hop)
    ham = sparse_ham(dev)
    print(ham.shape)
    print(ham.get_block(0, 0))




if __name__ == "__main__":
    main()
