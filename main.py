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
from cmpy.tightbinding import square_device, TbDevice, s_basis, sp3_basis
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

def flatten(i, j, data):
    shape = data.shape
    indices = np.indices(shape).reshape(2, np.prod(shape))
    indices += np.array([i, j])[:, np.newaxis]
    rows, cols = indices
    return rows, cols, data.flatten()


def coo_hamiltonian(device):
    rows, cols, data = list(), list(), list()
    for i, j, t_full in ham_data(device):
        r, c, t = flatten(i, j, t_full)
        rows += list(r)
        cols += list(c)
        data += list(t)
    n = device.lattice.n * device.n_orbs
    ham = coo_matrix((data, (rows, cols)), shape=(n, n))

    return ham





def main():
    print("Building lattice...")
    b = sp3_basis()
    dev = TbDevice.square((200, 16), eps=b.eps, t=b.hop)
    print("done")
    coo_ham = coo_hamiltonian(dev)
    print_mem()

if __name__ == "__main__":
    main()
