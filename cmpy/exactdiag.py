# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2022, Dylan Jones

import logging
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
import gftool as gt
from numba import njit, prange
from .basis import Sector, UP
from .matrix import EigenState
from .linalg import expm_multiply
from .models import AbstractManyBodyModel
from .operators import CreationOperator, AnnihilationOperator

logger = logging.getLogger(__name__)


def compute_groundstate(model, thresh=50):
    gs = EigenState()
    logger.debug("Computing ground-state:")
    for n_up, n_dn in model.basis.iter_fillings():
        hamop = model.hamilton_operator(n_up=n_up, n_dn=n_dn, dtype=np.float64)
        logger.debug("Sector (%d, %d), size: %d", n_up, n_dn, hamop.shape[0])

        if hamop.shape[0] <= thresh:
            ham = hamop.toarray()
            energies, vectors = np.linalg.eigh(ham)
            idx = np.argmin(energies)
            energy, state = energies[idx], vectors[:, idx]
        else:
            energies, vectors = sla.eigsh(hamop, k=1, which="SA")
            energy, state = energies[0], vectors[:, 0]

        if energy < gs.energy:
            gs = EigenState(energy, state, n_up, n_dn)
            logger.debug("gs-energy: %.6f", gs.energy)
    return gs


def solve_sector(model: AbstractManyBodyModel, sector: Sector, cache: dict = None):
    sector_key = (sector.n_up, sector.n_dn)
    if cache is not None and sector_key in cache:
        logger.debug("Loading eig  %d, %d", sector.n_up, sector.n_dn)
        eigvals, eigvecs = cache[sector_key]
    else:
        logger.debug("Solving eig  %d, %d (%s)", sector.n_up, sector.n_dn, sector.size)
        ham = model.hamiltonian(sector=sector)
        eigvals, eigvecs = np.linalg.eigh(ham)
        if cache is not None:
            cache[sector_key] = [eigvals, eigvecs]
    return eigvals, eigvecs


@njit(fastmath=True, nogil=True, parallel=True)
def occupation_up(up_states, dn_states, evals, evecs, beta, emin=0.0, pos=0):
    num_dn = len(dn_states)
    all_dn = np.arange(num_dn)
    occ = 0.0
    for up_idx in prange(len(up_states)):
        up = up_states[up_idx]
        if up & (1 << pos):  # state occupied
            indices = up_idx * num_dn + all_dn
            overlap = np.sum(np.abs(evecs[indices, :]) ** 2, axis=0)
            occ += np.sum(np.exp(-beta * (evals - emin)) * overlap)
    return occ


@njit(fastmath=True, nogil=True, parallel=True)
def occupation_dn(up_states, dn_states, evals, evecs, beta, emin=0.0, pos=0):
    num_dn = len(dn_states)
    all_up = np.arange(len(up_states))
    occ = 0.0
    for dn_idx in prange(num_dn):
        dn = dn_states[dn_idx]
        if dn & (1 << pos):  # state occupied
            indices = all_up * num_dn + dn_idx
            overlap = np.sum(np.abs(evecs[indices, :]) ** 2, axis=0)
            occ += np.sum(np.exp(-beta * (evals - emin)) * overlap)
    return occ


def occupation(up_states, dn_states, evals, evecs, beta, emin=0.0, pos=0, sigma=UP):
    if sigma == UP:
        return occupation_up(up_states, dn_states, evals, evecs, beta, emin, pos)
    else:
        return occupation_dn(up_states, dn_states, evals, evecs, beta, emin, pos)


@njit(fastmath=True, nogil=True, parallel=True)
def double_occupation(up_states, dn_states, evals, evecs, beta, emin=0.0, pos=0):
    occ = 0.0
    idx = 0
    for up_idx in prange(len(up_states)):
        for dn_idx in range(len(dn_states)):
            up = up_states[up_idx]
            dn = dn_states[dn_idx]
            if up & dn & (1 << pos):
                overlap = np.abs(evecs[idx, :]) ** 2
                occ += np.sum(np.exp(-beta * (evals - emin)) * overlap)
            idx += 1
    return occ


@njit(fastmath=True, nogil=True, parallel=True)
def _accumulate_sum(gf, z, evals, evals_p1, evecs_p1, cdag_evec, beta, emin):
    overlap = np.abs(evecs_p1.T.conj() @ cdag_evec) ** 2

    if np.isfinite(beta):
        exp_evals = np.exp(-beta * (evals - emin))
        exp_evals_p1 = np.exp(-beta * (evals_p1 - emin))
    else:
        exp_evals = np.ones_like(evals)
        exp_evals_p1 = np.ones_like(evals_p1)

    num_m = len(evals_p1)
    num_n = len(evals)
    for m in prange(num_m):
        for n in range(num_n):
            eig_m = evals_p1[m]
            eig_n = evals[n]
            weights = exp_evals[n] + exp_evals_p1[m]
            gf += overlap[m, n] * weights / (z + eig_n - eig_m)


def accumulate_gf(gf, z, cdag, evals, evecs, evals_p1, evecs_p1, beta, emin=0.0):
    cdag_evec = cdag.matmat(evecs)
    return _accumulate_sum(gf, z, evals, evals_p1, evecs_p1, cdag_evec, beta, emin)


class GreensFunctionMeasurement:
    def __init__(self, z, beta, pos=0, sigma=UP, dtype=None):
        self.z = z
        self.beta = beta
        self.pos = pos
        self.sigma = sigma
        self._part = 0
        self._gs_energy = np.infty
        self._gf = np.zeros_like(z, dtype=dtype)
        self._occ = 0.0
        self._occ_double = 0.0

    @property
    def part(self):
        return self._part * np.exp(-self.beta * self._gs_energy)

    @property
    def gf(self):
        return self._gf / self._part

    @property
    def occ(self):
        return self._occ / self._part

    @property
    def occ_double(self):
        return self._occ_double / self._part

    @property
    def gs_energy(self):
        return self._gs_energy

    def _acc_part(self, eigvals, factor=1.0):
        self._part *= factor
        self._part += np.sum(np.exp(-self.beta * (eigvals - self._gs_energy)))

    def _acc_gf(self, sector, sector_p1, evals, evecs, evals_p1, evecs_p1, factor):
        if factor != 1.0:
            self._gf *= factor

        cdag = CreationOperator(sector, sector_p1, pos=self.pos, sigma=self.sigma)
        z = self.z
        beta = self.beta
        e0 = self._gs_energy
        accumulate_gf(self._gf, z, cdag, evals, evecs, evals_p1, evecs_p1, beta, e0)

    def _acc_occ(self, sector, evals, evecs, factor):
        up = sector.up_states
        dn = sector.dn_states
        beta = self.beta
        e0 = self._gs_energy
        self._occ *= factor
        self._occ += occupation(up, dn, evals, evecs, beta, e0, self.pos, self.sigma)

    def _acc_occ_double(self, sector, evals, evecs, factor):
        up = sector.up_states
        dn = sector.dn_states
        beta = self.beta
        e0 = self._gs_energy
        self._occ_double *= factor
        self._occ_double += double_occupation(up, dn, evals, evecs, beta, e0, self.pos)

    def accumulate(self, sector, sector_p1, evals, evecs, evals_p1, evecs_p1):
        min_energy = min(evals)
        factor = 1.0
        if min_energy < self._gs_energy:
            factor = np.exp(-self.beta * (self._gs_energy - min_energy))
            self._gs_energy = min_energy
            logger.debug("New ground state: E_0=%.4f", min_energy)

        logger.debug("accumulating")
        self._acc_part(evals, factor)
        self._acc_gf(sector, sector_p1, evals, evecs, evals_p1, evecs_p1, factor)
        # self._acc_occ(sector, evals, evecs, factor)
        # self._acc_occ_double(sector, evals, evecs, factor)


def greens_function_lehmann(model, z, beta, pos=0, sigma=UP, eig_cache=None):
    logger.debug("Accumulating Lehmann sum (pos=%s, sigma=%s)", pos, sigma)
    data = GreensFunctionMeasurement(z, beta, pos, sigma)
    eig_cache = eig_cache if eig_cache is not None else dict()
    for n_up, n_dn in model.iter_fillings():
        sector = model.get_sector(n_up, n_dn)
        sector_p1 = model.basis.upper_sector(n_up, n_dn, sigma)
        if sector_p1 is not None:
            eigvals, eigvecs = solve_sector(model, sector, cache=eig_cache)
            eigvals_p1, eigvecs_p1 = solve_sector(model, sector_p1, cache=eig_cache)
            data.accumulate(sector, sector_p1, eigvals, eigvecs, eigvals_p1, eigvecs_p1)
        # else: eig_cache.clear()

    logger.debug("-" * 40)
    logger.debug("gs-energy:  %+.4f", data.gs_energy)
    logger.debug("occupation:  %.4f", data.occ)
    logger.debug("double-occ:  %.4f", data.occ_double)
    logger.debug("-" * 40)
    return data


def greens_greater(model, gs, start, stop, num=1000, pos=0, sigma=UP):
    n_up, n_dn = gs.n_up, gs.n_dn
    sector = model.basis.get_sector(n_up, n_dn)
    logger.debug("Computing greater GF (Sector: %d, %d; num: %d)", n_up, n_dn, num)

    times, dt = np.linspace(start, stop, num, retstep=True)
    sector_p1 = model.basis.upper_sector(n_up, n_dn, sigma)
    if sector_p1 is None:
        logger.warning("Upper sector not found!")
        return times, np.zeros_like(times)

    cop_dag = CreationOperator(sector, sector_p1, pos=pos, sigma=sigma)
    top_ket = cop_dag.matvec(gs.state)  # T|gs>
    bra_top = top_ket.conj()  # <gs|T

    hamop = -1j * model.hamilton_operator(sector=sector_p1)
    top_e0 = np.exp(+1j * gs.energy * dt)

    overlaps = expm_multiply(hamop, top_ket, start=start, stop=stop, num=num) @ bra_top

    factor = -1j * np.exp(+1j * gs.energy * times[0])
    overlaps[0] *= factor
    for n in range(1, num):
        factor *= top_e0
        overlaps[n] *= factor
    return times, overlaps


def greens_lesser(model, gs, start, stop, num=1000, pos=0, sigma=UP):
    n_up, n_dn = gs.n_up, gs.n_dn
    sector = model.basis.get_sector(n_up, n_dn)
    logger.debug("Computing lesser GF (Sector: %d, %d; num: %d)", n_up, n_dn, num)

    times, dt = np.linspace(start, stop, num, retstep=True)
    sector_m1 = model.basis.lower_sector(n_up, n_dn, sigma)
    if sector_m1 is None:
        logger.warning("Lower sector not found!")
        return times, np.zeros_like(times)

    cop = AnnihilationOperator(sector, sector_m1, pos=pos, sigma=sigma)
    top_ket = cop.matvec(gs.state)  # T|gs>
    bra_top = top_ket.conj()  # <gs|T

    hamop = +1j * model.hamilton_operator(sector=sector_m1)
    top_e0 = np.exp(-1j * gs.energy * dt)

    overlaps = expm_multiply(hamop, top_ket, start=start, stop=stop, num=num) @ bra_top

    factor = +1j * np.exp(-1j * gs.energy * times[0])
    overlaps[0] *= factor
    for n in range(1, num):
        factor *= top_e0
        overlaps[n] *= factor
    return times, overlaps


def greens_function_tevo(model, start, stop, num=1000, pos=0, sigma=UP):
    gs = compute_groundstate(model)
    times, gf_greater = greens_greater(model, gs, start, stop, num, pos, sigma)
    times, gf_lesser = greens_lesser(model, gs, start, stop, num, pos, sigma)
    return times, gf_greater - gf_lesser


def fourier_t2z(times, gf_t, omegas, delta=1e-2, eta=None):
    if eta is None:
        eta = -np.log(delta) / times[-1]
    z = omegas + 1j * eta
    gf_w = gt.fourier.tt2z(times, gf_t, z)
    return z, gf_w


# =========================================================================
# Lanczos diagonalization
# =========================================================================


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
        b = np.sqrt(b2)
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
