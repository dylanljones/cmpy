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
from cmpy import square_lattice, eta, plot_transmission, format_num
from cmpy.core.greens import greens
from cmpy.tightbinding import TbDevice, s_basis, sp3_basis
from scipy.sparse import coo_matrix


POINTS = [0, 0], [np.pi, 0], [np.pi, np.pi]
PNAMES = r"$\Gamma$", r"$X$", r"$M$"



def print_mem():
    vmem = psutil.virtual_memory()
    used = format_num(vmem.used)
    free = format_num(vmem.free)
    total = format_num(vmem.total)
    print(f"Free: {free}, Used: {used}, Total: {total}")


def main():
    model = TbDevice.square((5, 1))
    model.set_disorder(0.5)
    omegas = np.linspace(-5, 5, 100) + eta
    trans = model.transmission_curve(omegas)

    plot_transmission(omegas, trans)




if __name__ == "__main__":
    main()
