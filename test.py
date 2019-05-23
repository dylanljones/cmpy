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
    return f"{value:.{dec}f}+-{error:.{dec}f}"


def test_loclen():
    model = TbDevice.square((5, 1))
    model.set_disorder(1)

    lengths = np.arange(200, 300, 5)
    trans = model.transmission_loss(lengths, flatten=False, n_avrg=1000)
    trans = np.mean(np.log(trans), axis=1)

    ll, llerr, fit_data = loc_length_fit(lengths, trans)

    print(f"L=" + resultstring(ll, llerr))

    plot = Plot()
    plot.plot(*fit_data, ls="--", color="k", label=r"$\xi=" + f"{ll:.2f} \pm {llerr:.2f}$")
    plot.plot(lengths, trans)
    plot.legend()

    plot.show()


def main():
    basis = p3_basis(soc=1, ordering="orb")
    states = basis.states
    print(states)
    basis.show()


if __name__ == "__main__":
    main()
