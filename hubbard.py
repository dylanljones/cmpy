# -*- coding: utf-8 -*-
"""
Created on 29 Mar 2019
author: Dylan

project: cmpy2
version: 1.0
"""
import numpy as np
from scipy import linalg as la
from scipy.integrate import quad
from sciutils import Plot, eta
from cmpy import Hamiltonian, greens
from cmpy.hubbard import HubbardModel

t = 1
u = t * 10
mu = u / 2

w = np.sqrt((u/2)**2 + 4*t**2)
e0 = u/2 - w

# =============================================================================


def main():
    model = HubbardModel(eps=0, t=1, u=u, mu=mu)
    ham = model.hamiltonian(n=2)
    ham.show(ticklabels=model.state_labels)

    n = 100
    omegas = np.linspace(-10, 10, n) + eta
    gf = ham.gf(omegas[0])
    print(gf)
    # spec = -1/np.pi * np.sum(gf.imag, axis=1)
    # Plot.quickplot(omegas.real, spec)





if __name__ == "__main__":
    main()
