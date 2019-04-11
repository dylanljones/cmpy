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
import matplotlib.pyplot as plt
from cmpy import eta

t = 1
u = t * 10
mu = u / 2

w = np.sqrt((u/2)**2 + 4 * t**2)
e0 = u/2 - w


def spectral(g):
    return -1/np.pi * g.imag


def self_energy(omega, g, sign=1):
    return omega + mu + sign * t - (1/g)  # la.inv(g)


def gf(omega, sign=1):
    t1 = (0.5 + sign * (t / w)) / (omega + mu - (e0 + sign * t))
    t2 = (0.5 - sign * (t / w)) / (omega + mu - (u + sign * t - e0))
    return t1 + t2


def test_gf():
    n = 1000
    omegas = np.linspace(-10, 10, n) + eta

    gf_p = np.zeros(n, dtype="complex")
    gf_n = np.zeros(n, dtype="complex")
    for i in range(n):
        gf_p[i] = gf(omegas[i], +1)
        gf_n[i] = gf(omegas[i], -1)

    fig, ax = plt.subplots()

    col = "C0"
    spec = spectral(gf_p)
    ax.plot(omegas.real, spec / np.max(spec), label=r"$A (P=+1, k=0)$", color=col)
    sigma = - self_energy(omegas, gf_p, 1).imag
    ax.plot(omegas.real, sigma / np.max(sigma), label=r"$\Sigma$", color=col, ls="--")

    col = "C1"
    spec = spectral(gf_n)
    ax.plot(omegas.real, spec / np.max(spec), label=r"$A (P=-1, k=\pi)$", color=col)
    sigma = - self_energy(omegas, gf_n, -1).imag
    ax.plot(omegas.real, sigma / np.max(sigma), label=r"$\Sigma$", color=col, ls="--")

    plt.legend()
    plt.show()

# ================================================================


def rho(e):
    return np.sqrt(4 * t**2 - e**2) / (2 * np.pi * t**2)


def int_func(e, omega, sigma_latt):
    return rho(e) / (omega + mu - e - sigma_latt)


def gf_lattice(omega, sigma, a=-np.inf, b=np.inf):
    res = quad(int_func, a, b, args=(omega, sigma))
    return res[0]


def test_int():
    n = 100
    sigma = 0
    omegas = np.linspace(-10, 10, n)
    gf_latt = np.zeros(n)
    for i in range(n):
        gf_latt[i] = gf_lattice(omegas[i], sigma)
    plt.plot(omegas.real, gf_latt)
    plt.show()


def main():
    print(gf_lattice(0, 1))
    test_int()


if __name__ == "__main__":
    main()
