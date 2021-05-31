# coding: utf-8
#
# This code is part of cmpy.
# 
# Copyright (c) 2021, Dylan Jones

import logging
import numpy as np
from itertools import product
import scipy.sparse.linalg as sla
import gftool as gt
from .operators import CreationOperator, AnnihilationOperator, project_up, project_dn
from .matrix import EigenState
from .models import AbstractManyBodyModel
from .basis import Sector, UP
from .linalg import expm_multiply

logger = logging.getLogger(__name__)


def solve_sector(model: AbstractManyBodyModel, sector: Sector, cache: dict = None):
    sector_key = (sector.n_up, sector.n_dn)
    if cache is not None and sector_key in cache:
        logger.debug("Loading eig  %d, %d", sector.n_up, sector.n_dn)
        eigvals, eigvecs = cache[sector_key]
    else:
        logger.debug("Solving eig  %d, %d", sector.n_up, sector.n_dn)
        ham = model.hamilton_operator(sector=sector).array()
        eigvals, eigvecs = np.linalg.eigh(ham)
        if cache is not None:
            cache[sector_key] = [eigvals, eigvecs]
    return eigvals, eigvecs


def compute_groundstate(model, thresh=50):
    gs = EigenState()
    logger.debug("Computing ground-state:")
    for n_up, n_dn in model.basis.iter_fillings():
        hamop = model.hamilton_operator(n_up=n_up, n_dn=n_dn, dtype=np.float64)
        logger.debug(f"Sector (%d, %d), size: %d", n_up, n_dn, hamop.shape[0])

        if hamop.shape[0] <= thresh:
            ham = hamop.array()
            energies, vectors = np.linalg.eigh(ham)
            idx = np.argmin(energies)
            energy, state = energies[idx], vectors[:, idx]
        else:
            energies, vectors = sla.eigsh(hamop, k=1, which="SA")
            energy, state = energies[0], vectors[:, 0]

        if energy < gs.energy:
            gs = EigenState(energy, state, n_up, n_dn)
            logger.debug(f"gs-energy: %.6f", gs.energy)
    return gs


def occupation_up(sector, eigvals, eigvecs, beta, min_energy=0., pos=0):
    occ = 0.
    for up_idx, up in enumerate(sector.up_states):
        if up & (1 << pos):  # state occupied
            indices = project_up(up_idx, sector.num_dn, np.arange(sector.num_dn))
            overlap = np.sum(abs(eigvecs[indices, :])**2, axis=0)
            occ += np.sum(np.exp(-beta * (eigvals - min_energy)) * overlap)
    return occ


def occupation_dn(sector, eigvals, eigvecs, beta, min_energy=0., pos=0):
    occ = 0.
    for dn_idx, dn in enumerate(sector.dn_states):
        if dn & (1 << pos):  # state occupied
            indices = project_dn(dn_idx, sector.num_dn, np.arange(sector.num_up))
            overlap = np.sum(abs(eigvecs[indices, :])**2, axis=0)
            occ += np.sum(np.exp(-beta * (eigvals - min_energy)) * overlap)
    return occ


def occupation(sector, eigvals, eigvecs, beta, min_energy=0., pos=0, sigma=UP):
    if sigma == UP:
        return occupation_up(sector, eigvals, eigvecs, beta, min_energy, pos)
    return occupation_dn(sector, eigvals, eigvecs, beta, min_energy, pos)


def double_occupation(sector, eigvals, eigvecs, beta, min_energy=0., pos=0):
    occ = 0.
    for idx, (up, dn) in enumerate(product(sector.up_states, sector.dn_states)):
        if up & dn & (1 << pos):
            overlap = abs(eigvecs[idx, :]) ** 2
            occ += np.sum(np.exp(-beta * (eigvals - min_energy)) * overlap)
    return occ


def accumulate_gf(gf, z, cdag, eigvals, eigvecs, eigvals_p1, eigvecs_p1, beta, min_energy=0.):
    cdag_vec = cdag.matmat(eigvecs)
    overlap = abs(eigvecs_p1.T.conj() @ cdag_vec) ** 2

    if np.isfinite(beta):
        exp_eigvals = np.exp(-beta * (eigvals - min_energy))
        exp_eigvals_p1 = np.exp(-beta * (eigvals_p1 - min_energy))
    else:
        exp_eigvals = np.ones_like(eigvals)
        exp_eigvals_p1 = np.ones_like(eigvals_p1)

    for m, eig_m in enumerate(eigvals_p1):
        for n, eig_n in enumerate(eigvals):
            weights = exp_eigvals[n] + exp_eigvals_p1[m]
            gf += overlap[m, n] * weights / (z + eig_n - eig_m)


class GreensFunctionMeasurement:

    def __init__(self, z, beta, pos=0, sigma=UP, dtype=None):
        self.z = z
        self.beta = beta
        self.pos = pos
        self.sigma = sigma
        self._part = 0
        self._gs_energy = np.infty
        self._gf = np.zeros_like(z, dtype=dtype)
        self._occ = 0.
        self._occ_double = 0.

    @property
    def part(self):
        return self._part  * np.exp(-self.beta*self._gs_energy)

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

    def _acc_part(self, eigvals, factor=1.):
        self._part *= factor
        self._part += np.sum(np.exp(-self.beta * (eigvals - self._gs_energy)))

    def _acc_gf(self, sector, sector_p1, eigvals, eigvecs, eigvals_p1, eigvecs_p1, factor):
        if factor != 1.:
            self._gf *= factor

        cdag = CreationOperator(sector, sector_p1, pos=self.pos, sigma=self.sigma)
        z = self.z
        beta = self.beta
        e0 = self._gs_energy
        accumulate_gf(self._gf, z, cdag, eigvals, eigvecs, eigvals_p1, eigvecs_p1, beta, e0)

    def _acc_occ(self, sector, eigvals, eigvecs, factor):
        self._occ *= factor
        self._occ += occupation(sector, eigvals, eigvecs, self.beta, self._gs_energy,
                                self.pos, self.sigma)

    def _acc_occ_double(self, sector, eigvals, eigvecs, factor):
        self._occ_double *= factor
        self._occ_double += double_occupation(sector, eigvals, eigvecs, self.beta,
                                              self._gs_energy, self.pos)

    def _accumulate(self, sector, sector_p1, eigvals, eigvecs, eigvals_p1, eigvecs_p1, factor):
        self._acc_gf(sector, sector_p1, eigvals, eigvecs, eigvals_p1, eigvecs_p1, factor)
        self._acc_occ(sector, eigvals, eigvecs, factor)
        self._acc_occ_double(sector, eigvals, eigvecs, factor)

    def accumulate(self, sector, sector_p1, eigvals, eigvecs, eigvals_p1, eigvecs_p1):
        min_energy = min(eigvals)
        factor = 1
        if min_energy < self._gs_energy:
            factor = np.exp(-self.beta * (self._gs_energy - min_energy))
            self._gs_energy = min_energy
            logger.debug("Found new ground state energy: E_0=%.4f", min_energy)

        self._acc_part(eigvals, factor)
        self._accumulate(sector, sector_p1, eigvals, eigvecs, eigvals_p1, eigvecs_p1, factor)


def greens_function_lehmann(model, z, beta, pos=0, sigma=UP):
    logger.debug(f"Accumulating lehmann GF")
    data = GreensFunctionMeasurement(z, beta, pos, sigma)
    eig_cache = dict()
    for n_up, n_dn in model.iter_fillings():
        sector = model.get_sector(n_up, n_dn)
        sector_p1 = model.basis.upper_sector(n_up, n_dn, sigma)
        if sector_p1 is not None:
            eigvals, eigvecs = solve_sector(model, sector, cache=eig_cache)
            eigvals_p1, eigvecs_p1 = solve_sector(model, sector_p1, cache=eig_cache)
            data.accumulate(sector, sector_p1, eigvals, eigvecs, eigvals_p1, eigvecs_p1)
        else:
            eig_cache.clear()

    logger.debug("-"*40)
    logger.debug("gs-energy:  %+.4f", data.gs_energy)
    logger.debug("occupation:  %.4f", data.occ)
    logger.debug("double-occ:  %.4f", data.occ_double)
    logger.debug("-" * 40)
    return data


def greens_greater(model, gs, start, stop, num=1000, pos=0, sigma=UP):
    n_up, n_dn = gs.n_up, gs.n_dn
    sector = model.basis.get_sector(n_up, n_dn)
    logger.debug(f"Computing greater GF (Sector: %d, %d; num: %d)", n_up, n_dn, num)

    times, dt = np.linspace(start, stop, num, retstep=True)
    sector_p1 = model.basis.upper_sector(n_up, n_dn, sigma)
    if sector_p1 is None:
        logger.warning("Upper sector not found!")
        return times, np.zeros_like(times)

    cop_dag = CreationOperator(sector, sector_p1, pos=pos, sigma=sigma)
    top_ket = cop_dag.matvec(gs.state)  # T|gs>
    bra_top = top_ket.conj()            # <gs|T

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
    logger.debug(f"Computing lesser GF (Sector: %d, %d; num: %d)", n_up, n_dn, num)

    times, dt = np.linspace(start, stop, num, retstep=True)
    sector_m1 = model.basis.lower_sector(n_up, n_dn, sigma)
    if sector_m1 is None:
        logger.warning("Lower sector not found!")
        return times, np.zeros_like(times)

    cop = AnnihilationOperator(sector, sector_m1, pos=pos, sigma=sigma)
    top_ket = cop.matvec(gs.state)  # T|gs>
    bra_top = top_ket.conj()        # <gs|T

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


def fourier_t2z(times, gf_t, omegas, delta=1e-2):
    eta = -np.log(delta) / times[-1]
    z = omegas + 1j * eta
    gf_w = gt.fourier.tt2z(times, gf_t, z)
    return z, gf_w
