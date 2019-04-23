# -*- coding: utf-8 -*-
"""
Created on 26 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from cmpy.core import *
from cmpy.tightbinding import TbDevice
from cmpy.tightbinding.basis import *


points = [0, 0], [np.pi, 0], [np.pi, np.pi]
point_names = "$\Gamma$", "X", "M"


def show_bands(model):
    bands = model.bands(points)
    plot_bands(bands, point_names, show=False)


def plot_time(model, n_avrg=100):
    lengths = np.arange(50, 300, 25)
    n = len(lengths)
    times = np.zeros(n)
    tmp = np.zeros(n_avrg)
    for i in range(n):
        model.reshape(lengths[i])
        for j in prange(n_avrg):
            t0 = time.perf_counter()
            model.transmission()
            tmp[j] = time.perf_counter() - t0
        times[i] = np.mean(tmp)
    plt.plot(lengths, times)
    plt.show()

def main():
    model = TbDevice.square_p3((10, 8), eps_p=-2, soc=2)
    plot_time(model)
    return
    show_bands(model)

    omegas = np.linspace(-5, 5, 200) + eta
    trans = model.transmission_curve(omegas)
    plot_transmission(omegas, trans)

    return
    # model.set_disorder(1)
    t = model.mean_transmission()
    print(t)
    #trans = model.mean_transmission(n=1000)


if __name__ == "__main__":
    main()
