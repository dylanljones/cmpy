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
from abc import ABC
from typing import Union, Sequence
from scipy import sparse
from _expm_multiply import expm_multiply
from scipy.sparse.linalg import expm
from scipy.sparse.linalg import eigsh
from cmpy.basis import UP, EigenState, Sector
from cmpy.operators import (
    project_onsite_energy, project_interaction, project_site_hopping, CreationOperator,
    LinearOperator, AnnihilationOperator
)
from cmpy.greens import GreensFunction
from .abc import AbstractManyBodyModel


class HamiltonOperator(LinearOperator):

    def __init__(self, size, data, dtype=None):
        super().__init__((size, size), dtype=dtype)
        self.data = list(data)

    def _matvec(self, x):
        matvec = np.zeros_like(x)
        for row, col, val in self.data:
            # matvec[col] += val * x[row]
            matvec[row] += val * x[col]
        return matvec

    def _adjoint(self):
        return self

    def trace(self):
        return np.trace(self.array())

    def __rmul__(self, x):
        """Ensure trace-method in result."""
        scaled = super().__rmul__(x)
        scaled.trace = lambda: x*self.trace()
        return scaled


# =========================================================================
# Single impurity anderson model
# =========================================================================


class SingleImpurityAndersonModel(AbstractManyBodyModel, ABC):

    def __init__(self, u: Union[float, Sequence[float]] = 2.0,
                 eps_imp: Union[float, Sequence[float]] = 0.0,
                 eps_bath: Union[float, Sequence[float]] = 0.0,
                 v: Union[float, Sequence[float]] = 1.0,
                 mu: float = 0.0,
                 temp: float = 0.0):
        r"""Initializes the single impurity Anderson model

        Parameters
        ----------
        u : float
            The on-site interaction strength.
        eps_imp : float, optional
            The on-site energy of the impurity site. The default is `0`.
            Note that the impurity energy does not contain the chemical potential
        eps_bath : float or (N) float np.ndarray
            The on-site energy of the bath site(s). If a float is given the model
            is set to one bath site, otherwise the number of bath sites is given
            by the number of energy values passed. Note that the bath energy contains
            the chemical potential.
            If the SIAM is set to half filling, the bath energy can be fixed at
            .math:`\epsilon_B \mu = 0`. If `None`is given, one bath site at
            half filling will be set up.
            The default is `None` (half filling with one bath site).
        v : float or (N) float np.ndarray
            The hopping energy between the impurity site and the bath site(s).
            The number of hopping parameters must match the number of bath energies
            passed, i.e. the number of bath sites. The default is `1`.
        mu : float, optional
            The chemical potential of the system. If `None` the system is set
            to half filling, meaning a chemical potential of .math:`\mu = u/2`.
            The default is `None` (half filling).
        temp : float, optional
            Optional temperature in kelvin. The default is ``0``.
        """
        eps_bath = u / 2 if eps_bath is None else eps_bath
        mu = u / 2 if mu is None else mu
        eps_bath = np.atleast_1d(eps_bath).astype(np.float64)
        v = np.atleast_1d(v).astype(np.float64)
        num_sites = len(eps_bath) + 1
        super().__init__(num_sites, u=u, eps_imp=eps_imp, eps_bath=eps_bath, v=v, mu=mu, temp=temp)

    @property
    def num_bath(self) -> int:
        """The number of bath sites."""
        return len(self.eps_bath)

    @property
    def num_sites(self) -> int:
        """The total number of sites."""
        return self.num_bath + 1

    def update_bath_energy(self, eps_bath: (float, np.ndarray)) -> None:
        """Updates the on-site energies `eps_bath` of the bath sites.

        Parameters
        ----------
        eps_bath: float or (N) np.ndarray
            The energy values of the bath sites. If only one bath site
            is used a float value can be passed.
        """
        eps_bath = np.atleast_1d(eps_bath).astype(np.float64)
        if eps_bath.shape[0] != self.num_bath:
            raise ValueError(f"Dimension of the new bath energy {eps_bath.shape} "
                             f"does not match number of baths {self.num_bath}")
        self.eps_bath = eps_bath  # noqa

    def update_hybridization(self, v: (float, np.ndarray)) -> None:
        """Updates the hopping parameters `v` between the impurity and bath sites.

        Parameters
        ----------
        v: float or (N) np.ndarray
            The hopping parameters between the impurity and bath sites.
            If only one bath site is used a float value can be passed.
        """
        v = np.atleast_1d(v).astype(np.float64)
        if v.shape[0] != self.num_bath:
            raise ValueError(f"Dimension of the new hybridization {v.shape} "
                             f"does not match number of baths {self.num_bath}")
        self.v = v  # noqa

    def hybridization_func(self, z: np.ndarray) -> np.ndarray:
        r"""Computes the hybdridization function .math:`\Delta(z)` of the SIAM.

        The hybridization function of a single impurity Anderson model with
        .math:`N_b` bath sites is defined as
        ..math::
            \Delta(z) = \sum_{i}^{N_b} \frac{\abs{V_i}^2}{z - \epsilon_i}

        Parameters
        ----------
        z: (N) complex np.ndarray
            The complex frequency .math:`z`.

        Returns
        -------
        delta: (N) complex np.ndarray
        """
        x = z[..., np.newaxis]
        return np.sum(np.square(np.abs(self.v)) / (x - self.eps_bath), axis=-1)

    def hamiltonian_data(self, up_states, dn_states):
        num_sites = self.num_sites
        u = np.append(self.u, np.zeros(self.num_bath))
        eps = np.append(self.eps_imp, self.eps_bath) - self.mu
        hopping = lambda i, j: self.v[j - 1] if i == 0 else 0  # noqa

        yield from project_onsite_energy(up_states, dn_states, eps)
        yield from project_interaction(up_states, dn_states, u)
        yield from project_site_hopping(up_states, dn_states, num_sites, hopping, pos=0)

    def hamilton_operator(self, n_up=None, n_dn=None, sector=None, dtype=None):
        if sector is None:
            sector = self.basis.get_sector(n_up, n_dn)
        up_states, dn_states = sector.up_states, sector.dn_states
        size = len(up_states) * len(dn_states)

        data = list(self.hamiltonian_data(up_states, dn_states))
        return HamiltonOperator(size, list(data), dtype=dtype)

    def hamiltonian(self, n_up=None, n_dn=None, sector=None, dtype=None, dense=True):
        if sector is None:
            sector = self.basis.get_sector(n_up, n_dn)
        up_states, dn_states = sector.up_states, sector.dn_states
        size = len(up_states) * len(dn_states)

        rows, cols, data = list(), list(), list()
        for row, col, value in self.hamiltonian_data(up_states, dn_states):
            rows.append(row)
            cols.append(col)
            data.append(value)
        ham = sparse.csr_matrix((data, (rows, cols)), shape=(size, size))
        if dense:
            ham = ham.toarray()
        return ham

    def impurity_gf0(self, z):
        return 1 / (z + self.mu + self.eps_imp - self.hybridization_func(z))

    def impurity_gf(self, z, beta, sigma=UP):
        gf = GreensFunction(z, beta)
        part = 0
        eig_cache = dict()
        for n_up, n_dn in self.iter_fillings():
            # Solve particle sector
            sector_key = (n_up, n_dn)
            sector = self.get_sector(n_up, n_dn)
            if sector_key in eig_cache:
                eigvals, eigvecs = eig_cache[sector_key]
            else:
                ham = self.hamiltonian(sector=sector)
                eigvals, eigvecs = np.linalg.eigh(ham)
                eig_cache[sector_key] = [eigvals, eigvecs]

            # Update partition function
            part += np.sum(np.exp(-beta * eigvals))

            # Check if upper particle sector exists
            n_up_p1, n_dn_p1 = n_up, n_dn
            if sigma == UP:
                n_up_p1 += 1
            else:
                n_dn_p1 += 1

            if n_up_p1 in self.basis.fillings and n_dn_p1 in self.basis.fillings:
                # Solve upper particle sector
                sector_p1 = self.basis.get_sector(n_up_p1, n_dn_p1)
                ham_p1 = self.hamiltonian(sector=sector_p1)
                eigvals_p1, eigvecs_p1 = np.linalg.eigh(ham_p1)
                eig_cache[(n_up_p1, n_dn_p1)] = [eigvals_p1, eigvecs_p1]

                # Update Greens-function
                cdag = CreationOperator(sector, sector_p1, sigma=sigma)
                gf.accumulate(cdag, eigvals, eigvecs, eigvals_p1, eigvecs_p1)
            else:
                eig_cache = dict()
        return gf / part

    def get_sector_gs(self, sector):
        ham = self.hamiltonian(sector=sector)
        if ham.shape[0] < 50:
            eigvals, eigvecs = np.linalg.eigh(ham)
            gs_idx = np.argmin(eigvals)
            return eigvals[gs_idx], eigvecs[:, gs_idx], ham
        energy, vector = eigsh(ham, k=1, which="SA")
        return energy[0], vector[:,0], ham

    def get_total_gs(self):
        gs = EigenState(energy=np.infty, vector=None, nup=None, ndn=None, ham=None)
        for n_up, n_dn in self.iter_fillings():
            sector = self.get_sector(n_up, n_dn)
            energy, vector, ham = self.get_sector_gs(sector)
            if energy < gs.energy:
                gs = EigenState(energy=energy, vector=vector, nup=n_up, ndn=n_dn, ham=ham)
        return gs

    def t_evo_gr(self, gs: EigenState, start, stop, num):
        tt, dt = np.linspace(start, stop, num=num, retstep=True)
        if gs.nup + 1 > self.num_sites:
            return tt, np.zeros_like(tt)

        gs_sector = self.get_sector(gs.nup, gs.ndn)
        gs_up1_sector = self.get_sector(gs.nup+1, gs.ndn)
        cd_0_up = CreationOperator(gs_sector, gs_up1_sector, pos=0, sigma=UP)
        ket = cd_0_up.matvec(gs.vector)
        bra = ket.conj()
        print(bra, ket)

        tevo_gs_energy = np.exp(1j*gs.energy*dt)
        # exponential operator exp(-i*H*t):
        ham = self.hamilton_operator(sector=gs_up1_sector)
        tevo_op_exp = -1j * ham @ np.eye(*ham.shape) * dt
        # overlaps = expm_multiply(tevo_op, ket, start=start, stop=stop, num=num) @ bra
        overlaps = np.zeros(num, dtype=complex)
        # tevo_ket = expm(tevo_op_exp) @ ket
        tevo_op = expm(tevo_op_exp)

        factor = -1j
        overlaps[0] = factor*np.dot(bra, ket)
        for nn in range(1, num):
            factor *= tevo_gs_energy
            ket = tevo_op @ ket
            overlaps[nn] = np.dot(bra, ket) * factor
        return tt, overlaps

    def t_evo_ls(self, gs: EigenState, start, stop, num):
        tt, dt = np.linspace(start, stop, num=num, retstep=True)
        if gs.nup == 0:
            print("No up electron to annihilate.")
            return tt, np.zeros_like(tt)

        gs_sector = self.get_sector(gs.nup, gs.ndn)
        gs_upm1_sector = self.get_sector(gs.nup-1, gs.ndn)
        c_0_up = AnnihilationOperator(gs_sector, gs_upm1_sector, pos=0, sigma=UP)
        ket = c_0_up.matvec(gs.vector)
        bra = ket.conj()

        tevo_gs_energy = np.exp(-1j*gs.energy*dt)
        # exponential operator exp(-i*H*t):
        ham = self.hamilton_operator(sector=gs_upm1_sector)
        tevo_op_exp = -1j * ham @ np.eye(*ham.shape) * dt
        # overlaps = expm_multiply(tevo_op, ket, start=start, stop=stop, num=num) @ bra
        overlaps = np.zeros(num, dtype=complex)
        # tevo_ket = expm(tevo_op_exp) @ ket
        tevo_op = expm(tevo_op_exp)

        factor = -1j
        overlaps[0] = factor*np.dot(bra, ket)
        for nn in range(1, num):
            factor *= tevo_gs_energy
            ket = tevo_op @ ket
            overlaps[nn] = np.dot(bra, ket) * factor
        return tt, overlaps



    def pformat(self):
        return f"U={self.u}, ε_i={self.eps_imp}, ε_b={self.eps_bath}, v={self.v}, " \
               f"μ={self.mu}, T={self.temp}"
