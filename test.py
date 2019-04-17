# -*- coding: utf-8 -*-
"""
Created on 26 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
import os
import numpy as np
from cmpy.core import *
from cmpy.tightbinding import TbDevice


def estimate_max_memory(model, max_length):
    total_orbs = int(model.n_orbs * model.lattice.slice_sites * max_length)
    size = total_orbs**2 * int(np.dtype("complex").itemsize)
    print(format_num(size))


def test_transmission():
    model = TbDevice.square_p3((10, 1))
    model.set_disorder(1)
    lengths = np.arange(100, 200, 5)
    trans = model.transmission_loss(lengths, n_avrg=300)


def main():
    model = TbDevice.square_sp3((300, 10))
    ham = model.hamiltonian()
    print(format_num(ham.nbytes))


if __name__ == "__main__":
    main()
