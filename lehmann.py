# -*- coding: utf-8 -*-
"""
Created on 17 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from sciutils import Plot, use_cycler
from cmpy import get_omegas
from cmpy.models import Siam

use_cycler()


def main():
    u = 5
    mu = u/2
    eps_imp, eps_bath, v = 0, mu, 1
    omegas, eta = get_omegas(6, deta=0.5)
    z = omegas + eta

    siam = Siam(u, eps_imp, eps_bath, v, mu)
    # siam.set_basis([1, 2])
    siam.sort_states(key=lambda x: x.particles)
    siam.show_hamiltonian(False)

    gfu = siam.impurity_gf(z, 0)
    gfd = siam.impurity_gf(z, 1)

    plot = Plot()
    plot.plot(omegas, -gfu.imag, label=r"G$_{\uparrow}$")
    plot.plot(omegas, -gfd.imag, ls="--", label=r"G$_{\downarrow}$")
    plot.legend()
    plot.show()


if __name__ == "__main__":
    main()
