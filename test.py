# -*- coding: utf-8 -*-
"""
Created on 26 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
from cmpy import *
from cmpy.tightbinding import *

POINTS = [0, 0], [np.pi, 0], [np.pi, np.pi]
NAMES = r"$\Gamma$", r"$X$", r"$M$"

# =========================================================================


def resultstring(value, error, dec=2):
    return f"{value:.{dec}f}{Symbols.pm}{error:.{dec}f}"


def main():
    model = TbDevice.square((5, 1))
    model.set_disorder(1)

    lengths = np.arange(50, 200, 5)
    trans = model.transmission_loss(lengths, flatten=False, n_avrg=200)
    trans = np.mean(np.log(trans), axis=1)

    ll, llerr, fit_data = loc_length_fit(lengths, trans)

    print(f"{Symbols.Lambda}=" + resultstring(ll, llerr))

    plot = Plot()
    plot.plot(*fit_data, ls="--", color="k")
    plot.plot(lengths, trans)

    plot.show()


if __name__ == "__main__":
    main()
