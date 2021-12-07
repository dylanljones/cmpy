# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import numpy as np
from numpy.lib import scimath
import scipy.linalg as la


def iter_lanczos_coeffs(ham, size=10):
    # Initial guess of wavefunction
    psi = np.random.uniform(0, 1, size=len(ham))
    # First iteration only contains diagonal coefficient
    a = np.dot(psi, np.dot(ham, psi)) / np.dot(psi, psi)
    yield a, None
    # Compute new wavefunction:
    # |ψ_1⟩ = H |ψ_0⟩ - a_0 |ψ_0⟩
    psi_new = np.dot(ham, psi) - a * psi
    psi_prev = psi
    psi = psi_new
    # Continue iterations
    for n in range(1, size):
        # Compute coefficients a_n, b_n^2
        a = np.dot(psi, np.dot(ham, psi)) / np.dot(psi, psi)
        b2 = np.dot(psi, psi) / np.dot(psi_prev, psi_prev)
        # Compute new wavefunction
        # |ψ_{n+1}⟩ = H |ψ_n⟩ - a_n |ψ_n⟩ - b_n^2 |ψ_{n-1}⟩
        psi_new = np.dot(ham, psi) - a * psi - b2 * psi_prev
        # Save coefficients and update wave functions
        b = scimath.sqrt(b2)
        yield a, b
        psi_prev = psi
        psi = psi_new


def lanczos_coeffs(ham, size=10):
    a_coeffs = list()
    b_coeffs = list()
    for a, b in iter_lanczos_coeffs(ham, size):
        a_coeffs.append(a)
        b_coeffs.append(b)
    # remove None from b_coeffs
    b_coeffs.pop(0)
    return a_coeffs, b_coeffs


def lanczos_matrix(a_coeffs, b_coeffs):
    mat = np.diag(a_coeffs)
    np.fill_diagonal(mat[1:], b_coeffs)
    np.fill_diagonal(mat[:, 1:], b_coeffs)
    return mat


def lanczos_ground_state(a_coeffs, b_coeffs, max_eig=3):
    xi, vi = la.eigh_tridiagonal(
        a_coeffs, b_coeffs, select="i", select_range=(0, max_eig)
    )
    idx = np.argmin(xi)
    e_gs = xi[idx]
    gs = vi[:, idx]
    return e_gs, gs
