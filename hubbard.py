# -*- coding: utf-8 -*-
"""
Created on 29 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
import numpy as np
from scipy import linalg as la
from scipy.integrate import quad
from cmpy import eta, greens
from cmpy import Hamiltonian, Matrix, Plot, prange
from cmpy.hubbard import Basis, HubbardModel

t = 1
u = t * 10
mu = u / 2

w = np.sqrt((u/2)**2 + 4*t**2)
e0 = u/2 - w

# =============================================================================


def greens_function(omegas, ham):
    n = len(omegas)
    gf = np.zeros(n)
    for i in range(n):
        eigvals = ham.eigvals()

    return gf



def test_hubbard():
    omegas = np.linspace(-5, 5, 100)
    model = HubbardModel(eps=0, t=1, u=u, mu=mu)
    ham = model.hamiltonian(n=2, spin=0)

    eigvals, eigvecs = ham.eig()
    print(np.sort(eigvals.real))
    plot = ham.show(ticklabels=model.basis.state_latex_strings())
    plot.show()

    gf = greens_function(omegas + eta, ham)
    return
    plot = Plot()
    plot.plot(omegas, gf)
    plot.show()

    eigvals, eigvecs = ham.eig()
    print(eigvals)
    print(e0)


def gf_lehmann(omega, ham):
    eigvals = ham.eigvals()
    n = len(eigvals)
    gf = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                gf += 1 / (omega - (eigvals[i] - eigvals[j]))
    return gf


def test_gf():
    olim = -5, 5
    ham = Hamiltonian.zeros(3)
    ham.set_hopping(0, 1, 1)
    ham.set_hopping(1, 2, 1)

    n = 100
    omegas = np.linspace(*olim, n) + eta
    gf = np.zeros(n, dtype="complex")
    gf2 = np.zeros(n, dtype="complex")
    for i in range(n):
        gf[i] = np.trace(greens.greens(ham, omegas[i]))
        gf2[i] = gf_lehmann(omegas[i], ham)
    plot = Plot()
    plot.plot(omegas.real, -gf.imag)
    plot.plot(omegas.real, -gf2.imag)
    plot.show()

def main():
    test_gf()


if __name__ == "__main__":
    main()
