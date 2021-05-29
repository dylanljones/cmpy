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
from cmpy.basis import UP
from cmpy.operators import (
    project_onsite_energy, project_interaction, project_site_hopping, CreationOperator,
    LinearOperator
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
            matvec[col] += val * x[row]
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
                 mu: float = None,
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
            .math:`\epsilon_B \mu = 0`.
            The default is `0` (half filling with one bath site).
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
        mu = u / 2 if mu is None else mu
        eps_bath = np.atleast_1d(eps_bath)
        v = np.atleast_1d(v)
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
        eps = np.append(self.eps_imp - self.mu, self.eps_bath)
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

    def hamiltonian(self, n_up=None, n_dn=None, sector=None, dtype=None):
        hamop = self.hamilton_operator(n_up, n_dn, sector, dtype)
        return hamop.array()

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
                cop_dag = CreationOperator(sector, sector_p1, sigma=sigma)
                gf.accumulate(cop_dag, eigvals, eigvecs, eigvals_p1, eigvecs_p1)
            else:
                eig_cache = dict()
        return gf / part

    def pformat(self):
        return f"U={self.u}, ε_i={self.eps_imp}, ε_b={self.eps_bath}, v={self.v}, " \
               f"μ={self.mu}, T={self.temp}"
