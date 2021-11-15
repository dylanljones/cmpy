# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Coherent potential approximation."""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from functools import partial


def gf_cpa_loop(z, eps, con, hilbert, thresh=1e-5, maxiter=1000):
    sigma = np.zeros(len(z))
    sigma_old = np.inf
    gf_avrg = np.zeros_like(z)
    errs = list()
    for i in range(maxiter):
        # Compute average GF via the self-energy
        # <G> = G_0(E - Σ) = 1 / (E - H_0 - Σ)
        gf_avrg = hilbert(z - sigma)
        gf_avrg_inv = 1 / gf_avrg
        # gf0_inv = 1 / hilbert(z)
        gf0_inv = sigma + gf_avrg_inv

        # Dyson equation:
        # G_i = 1 / (E - H_0 - eps_i)
        gf_i = 1 / (sigma[..., np.newaxis] - eps + gf_avrg_inv[..., np.newaxis])
        gf_avrg = np.sum(con * gf_i, axis=-1)   # <G> = c_1 G_1 + ... + c_n G_n
        sigma = gf0_inv - 1 / gf_avrg           # Σ = G_0^{-1} - <G>^{-1}

        diff = np.trapz(abs(sigma - sigma_old), z.real)
        errs.append(diff)
        if diff < thresh:
            print(f"Converged in {i} iterations")
            break
        sigma_old = sigma
    else:
        print("CPA self consistency not reached")

    return gf_avrg


def sigma_root(sigma, z, eps, con, hilbert):
    # GF of effective medium
    gf_0 = hilbert(z - sigma)
    # T-matrix elements
    eps_diff = eps - sigma[..., np.newaxis]
    ti = eps_diff / (1 - eps_diff * gf_0[..., np.newaxis])
    # Average T-matrix
    tmat = np.sum(con * ti, axis=-1)
    # Self energy
    return tmat / (1 + tmat * gf_0)


def sigma_root_restricted(sigma, z, eps, con, hilbert):
    # Mask of invalid roots Im(Σ) > 0
    mask = sigma.imag > 0

    if np.all(~mask):
        # All sigmas valid
        return sigma_root(sigma, z, eps, con, hilbert)

    # Store offset to valid solution and remove invalid
    offset = sigma.imag[mask].copy()
    sigma.imag[mask] = 0

    # Compute root equation and enlarge residues
    root = np.asarray(sigma_root(sigma, z, eps, con, hilbert))
    root[mask] *= (1 + offset)

    # Remove invalid roots
    root.real[mask] += 1e-3 * offset * np.where(root.real[mask] >= 0, 1, -1)
    root.real[mask] += 1e-3 * offset * np.where(root.real[mask] >= 0, 1, -1)

    return root


def solve_sigma_root(z, eps, con, hilbert_trans, restricted=True, **kwargs):
    con = np.array(con)
    eps = np.array(eps)

    # Initial guess
    sigma = np.sum(eps * con, axis=-1).astype(np.complex256)
    sigma, __ = np.broadcast_arrays(sigma, z)

    # Setup arguments
    func = sigma_root_restricted if restricted else sigma_root
    root_eq = partial(func, z=z, eps=eps, con=con, hilbert=hilbert_trans)
    kwargs.setdefault("method", "anderson" if restricted else "broyden2")

    # Optimize root
    sol = optimize.root(root_eq, x0=sigma, **kwargs)
    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.x


def gf_cpa_solve_root(z, eps, con, hilbert, restricted=True, **kwargs):
    sigma = solve_sigma_root(z, eps, con, hilbert, restricted, **kwargs)
    return hilbert(z - sigma)
