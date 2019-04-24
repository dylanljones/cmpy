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
from cmpy.tightbinding import TbDevice, loc_length
from cmpy.tightbinding.basis import *


points = [0, 0], [np.pi, 0], [np.pi, np.pi]
point_names = "$\Gamma$", "X", "M"


def show_bands(model):
    bands = model.bands(points)
    plot_bands(bands, point_names, show=False)


def test_disorder(model, disorder):
    model.set_disorder(disorder)

    lengths = np.arange(50, 200, 10)

    t = model.transmission_loss(lengths, n_avrg=1000, flatten=True)
    plot_transmission_loss(lengths, t)
    print(loc_length(lengths, np.log10(t)))


def test_soc_strength(soc):
    w = 2
    lengths = np.arange(25, 100, 25)
    plot = Plot()

    for soc in range(0, 11, 1):
        model = TbDevice.square_p3((200, 4), soc=soc)
        model.set_disorder(w)
        t = model.transmission_loss(lengths, n_avrg=100, flatten=True)
        plot.plot(lengths, np.log10(t), label=r"$\lambda_{SOC}=$" + str(soc))
        print(loc_length(lengths, np.log10(t)))
        print()
    plot.legend()
    plot.show()


def test_logarithmic(h=1, soc=5):
    model = TbDevice.square_p3((200, h), soc=soc)
    model.set_disorder(w)



def main():
    model = TbDevice.square_p3((200, 1), eps_p=0, soc=1)
    #show_bands(model)

    test_disorder(model, 20)
    # test_disorder(model, 0.5)

    #omegas = np.linspace(-5, 5, 200) + eta
    #trans = model.transmission_curve(omegas)
    #plot_transmission(omegas, trans)


if __name__ == "__main__":
    main()
