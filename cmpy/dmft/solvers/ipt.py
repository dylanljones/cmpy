# -*- coding: utf-8 -*-
"""
Created on 14 Aug 2019
author: Dylan Jones

Iterative Pertubative Theory (IPT) solvers

project: cmpy
version: 1.0
"""
import numpy as np
import scipy.signal as signal
from cmpy.core import bethe_gf_omega, gf_omega_fft, gf_tau_fft, fermi_dist
from .utils import time_freq_arrays


def _ph_hf_sigma(spec, nf, u):
    """Imaginary part of the second order diagram
    because of particle-hole symmetry at half-fill in the Single band
    one can work with A^+ only"""
    ap = spec * nf
    # convolution A^+ * A^+
    app = signal.fftconvolve(ap, ap, mode='same')
    # convolution A^-(-w) * App
    appp = signal.fftconvolve(ap, app, mode='same')
    return -np.pi * u**2 * (appp + appp[::-1])


class IptSolver:

    def __init__(self, u, t, omega, tau=None, nf=None):
        self.omega = omega
        self.dw = omega[1] - omega[0]
        self.tau = tau
        self.t = t
        self.u = u
        self.nf = nf

    def solve_real(self, gf_bath):
        spec = -1 / np.pi * gf_bath.imag
        # Second order diagram
        isi = _ph_hf_sigma(spec, self.nf, self.u) * self.dw ** 2
        isi = 0.5 * (isi + isi[::-1])
        # Kramers-Kronig relation, uses Fourier Transform to speed convolution
        hsi = -signal.hilbert(isi, len(isi) * 4)[:len(isi)].imag
        sigma = hsi + 1j * isi
        # Hilbert transform on bethe lattice
        gf = bethe_gf_omega(self.omega - sigma)
        return gf, sigma

    def solve_imag(self, gf_bath):
        gf_tau = gf_omega_fft(gf_bath, self.tau, self.omega, [1., 0., 0.25])
        sigma_tau = self.u**2 * gf_tau**3
        sigma_omega = gf_tau_fft(sigma_tau, self.tau, self.omega, [self.u**2/4, 0, 0])
        gf_imp = gf_bath/(1-sigma_omega*gf_bath)
        return gf_imp, sigma_omega


# =========================================================================


def imag_loop(beta, u, t, n=1000, mix=1., thresh=1e-3, max_iter=10000):
    tau, omega = time_freq_arrays(beta, n)
    # eta = 2j * (omega[1] - omega[0])
    iomega = omega * 1j
    solver = IptSolver(u, t, omega, tau=tau)
    gf = bethe_gf_omega(iomega)
    sigma = 0
    for _ in range(max_iter):
        gf_old = gf.copy()
        gf_bath = 1 / (iomega - t**2 * gf_old)
        gf_new, sigma = solver.solve_imag(gf_bath)
        gf_new.real = 0.
        converged = np.allclose(gf_old, gf_new, thresh)
        if converged:
            break
        gf = mix * gf_new + (1 - mix) * gf_old
    return omega, gf, sigma


def real_loop(omega, beta, u, t=0.5, mix=1., thresh=1e-10, max_iter=10000):
    eta = 2j * (omega[1] - omega[0])
    nf = fermi_dist(omega, beta, mu=0)

    solver = IptSolver(u, t, omega, nf=nf)
    gf = bethe_gf_omega(omega + eta)
    sigma = 0
    for i in range(max_iter):
        gf_old = gf.copy()
        gf_bath = 1 / (omega + eta - t**2 * gf)

        gf_new, sigma = solver.solve_real(gf_bath)
        converged = np.allclose(gf_new, gf_old, atol=thresh)
        if converged:
            break
        gf = mix * gf_new + (1 - mix) * gf_old
    return gf, sigma
