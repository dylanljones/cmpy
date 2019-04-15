# -*- coding: utf-8 -*-
"""
Created on 26 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
import numpy as np
from cmpy.core import *
from cmpy.tightbinding import TbDevice


def estimate_max_memory(model, max_length):
    total_orbs = int(model.n_orbs * model.lattice.slice_sites * max_length)
    size = total_orbs**2 * int(np.dtype("complex").itemsize)
    print(format_num(size))


def main():
    model = TbDevice.square((10, 1))
    model.set_disorder(1)

    lengths = np.arange(100, 200, 10)
    trans = model.transmission_loss(lengths, n_avrg=1000, flatten=True)
    plot_transmission_loss(lengths, trans)
    # estimate_max_memory(model, lengths[-1])

    # trans = model.transmission_loss(lengths, n_avrg=100, flatten=True)
    # plot_transmission_loss(lengths, trans)


if __name__ == "__main__":
    main()
