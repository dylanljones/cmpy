# -*- coding: utf-8 -*-
"""
Created on 29 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
import time
import numpy as np
from cmpy import square_lattice, time_str
from cmpy import plot_transmission, eta, plot_transmission_loss
from cmpy.tightbinding import TbDevice, sp3_basis


def main():
    t0 = time.perf_counter()
    latt = square_lattice(shape=(1000, 32))
    t = time.perf_counter() - t0
    print(f"Total time: {time_str(t)}")


if __name__ == "__main__":
    main()
