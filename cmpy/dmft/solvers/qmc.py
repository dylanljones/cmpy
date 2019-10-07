# -*- coding: utf-8 -*-
"""
Created on 14 Aug 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from scitools import Matrix, Plot
from cmpy.core import gf_omega_fft, gf_tau_fft, gf_lehmann, bethe_gf_omega

T = 0.2


def interaction_matrix(n_bands=1):
    particles = 2 * n_bands
    fields = 2 * particles - 1
    matrix = Matrix.zeros(particles, fields)
    inter = 0
    for i in range(particles):
        for j in range(i+1, particles):
            matrix[i, inter] = 1
            matrix[j, inter] = -1
            inter += 1
    return matrix


def ising_v(dtau, u, length, fields=1, polar=0.5):
    lam = np.arccosh(np.exp(dtau * u/2))
    v = np.ones((fields, length))
    rand = np.random.rand(fields, length)
    v[rand > polar] = -1
    return lam * v


def gf_met(omega, mu, sigma_p, sigma=0.5, sigma_n=0):
    """Double semi-circular density of states to represent the non-interacting
    dimer """
    g_1 = bethe_gf_omega(omega + mu - sigma_p)
    g_2 = bethe_gf_omega(omega + mu + sigma_p)
    g_d = .5 * (g_1 + g_2)
    g_o = .5 * (g_1 - g_2)
    return g_d, g_o


def gf_tail(gtau, u, mu, tp):

    g_t0 = gtau[:, :, 0]
    gtail = [np.eye(2).reshape(2, 2, 1),
             (-mu - ((u - .5 * tp) * (0.5 + g_t0)) * np.eye(2) +
              tp * (1 - g_t0) * np.array([[0, 1], [1, 0]])).reshape(2, 2, 1),
             (0.25 + u**2 / 4 + tp**2) * np.eye(2).reshape(2, 2, 1)]
    return gtail


def main():
    n = 256
    n_orbs = 1
    u = 10
    mu = 0
    tp = 1
    beta = 1/T

    dtau = beta / n
    tau, freq = time_freq_arrays(beta, n)
    omega = freq + 0.001j
    intm = interaction_matrix(n_orbs)
    v_field = ising_v(dtau, u, length=2*n)
    giw_d, giw_o = gf_met(omega, mu, tp)
    gmix = np.array([[omega + mu, -tp * np.ones_like(freq)],
                     [-tp * np.ones_like(freq), omega + mu]])
    giw = np.array([[giw_d, giw_o], [giw_o, giw_d]])
    g0tau0 = -0.5 * np.eye(2).reshape(2, 2, 1)
    gtau = gf_omega_fft(giw, tau, freq, gf_tail(g0tau0, 0., mu, tp))
    print(gtau.shape)

    plot = Plot()
    plot.plot(gtau[0, 0, :].imag)
    plot.show()


if __name__ == "__main__":
    main()
