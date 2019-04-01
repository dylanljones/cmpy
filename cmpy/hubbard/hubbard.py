# -*- coding: utf-8 -*-
"""
Created on 29 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
import numpy as np
import matplotlib.pyplot as plt
from cmpy.core import Hamiltonian, Matrix, greens
from cmpy.core import square_lattice, eta
from .basis import Basis

t = 1
u = 10 * t
w = np.sqrt(u**2/4 + 4*t**2)
e0 = u/2 - w


class HubbardModel:

    def __init__(self, eps=0., t=1., u=None, mu=None):
        self.lattice = square_lattice()
        self.basis = None
        self.states = list()

        self.eps = eps
        self.t = t
        self.u = 10 * t if u is None else u
        self.mu = self.u / 2 if mu is None else mu

        self.build(x=2)

    def build(self, x=2, y=1):
        self.lattice.build((x, y))
        self.basis = Basis(int(x * y))

    def hamiltonian(self, n=None, spin=None):
        states = self.basis.get_states(n, spin)
        ham = Hamiltonian.zeros(len(states))
        for i, j in ham.iter_indices():
            state = states[i]
            if i == j:
                ham.set_energy(i, self.eps)
            if j < i:
                hops = state.check_hopping(states[j])
                if hops is not None:
                    i1, i2 = hops
                    if abs(i1 - i2) == 1:
                        ham.set_hopping(i, j, self.t)
        for i in range(ham.n):
            state = states[i]
            ham.set_energy(i, self.u * state.n_interaction())
        self.states = states
        return ham

    def __repr__(self):
        return f"Hubbard(\u03b5={self.eps}, t={self.t}, U={self.u}, \u03bc={self.mu})"


def eigvecs_1(u, t):
    return -np.sqrt((w + u/2)/(2*w)), -2*t/np.sqrt(2*w*(w + u/2))


def gf_test(omega, mu, par=1):
    x1 = 0.5 + par * t/w
    x2 = 0.5 - par * t/w
    d = omega + mu
    return x1/(d - (e0 + par * t)) + x2/(d - (u + par * t - e0))


def gf_lehman(model, eigvals, psi, omega):
    gf = 0
    n = len(eigvals)
    for i in range(n):
        for j in range(n):
            if i != j:
                gf += 1/(omega + model.mu - (eigvals[i] - eigvals[j]))
    return gf


def spectral(g):
    return -1/np.pi * g.imag


def main():
    model = HubbardModel()
    ham = model.hamiltonian(2, 0)
    ham.show(ticklabels=[x.repr for x in model.states])

    energy0, psi0 = ham.ground_state()
    print(energy0, psi0)

    eigvals, eigvecs = ham.eig()
    n = 1000
    omegas = np.linspace(-10, 10, n) + eta
    gf = np.zeros(n, "complex")
    for i in range(n):
        gf[i] = gf_lehman(model, eigvals, eigvecs, omegas[i])
    plt.plot(omegas.real, spectral(gf))
    plt.show()


if __name__ == "__main__":
    main()
