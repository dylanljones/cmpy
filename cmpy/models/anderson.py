# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2022, Dylan Jones

import numpy as np
from typing import Union, Sequence
from cmpy.basis import UP
from cmpy.operators import (
    project_onsite_energy,
    project_hubbard_inter,
    project_hopping,
)
from cmpy.exactdiag import greens_function_lehmann
from .abc import AbstractManyBodyModel


# =========================================================================
# Single impurity anderson model
# =========================================================================


class SingleImpurityAndersonModel(AbstractManyBodyModel):
    def __init__(
        self,
        u: Union[float, Sequence[float]] = 2.0,
        eps_imp: Union[float, Sequence[float]] = 0.0,
        eps_bath: Union[float, Sequence[float]] = 0.0,
        v: Union[float, Sequence[float]] = 1.0,
        mu: float = None,
        temp: float = 0.0,
    ):
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
            the chemical potential. If the SIAM is set to half filling, the bath energy
            can be fixed at .math:`\epsilon_B = 0`.
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
        num_bath = len(eps_bath)
        if num_bath > 1 and len(v) == 1:
            v = np.ones(len(eps_bath)) * v[0]
        if num_bath == 1 and len(v) > 1:
            eps_bath = np.ones(len(v)) * eps_bath[0]
        assert len(eps_bath) == len(v), (
            f"Shape of bath on-site energy {num_bath} "
            f"doesn't match hybridization {len(v)}!"
        )
        num_sites = len(eps_bath) + 1
        super().__init__(
            num_sites, u=u, eps_imp=eps_imp, eps_bath=eps_bath, v=v, mu=mu, temp=temp
        )

    @property
    def num_bath(self) -> int:
        """The number of bath sites."""
        return len(self.eps_bath)

    @property
    def num_sites(self) -> int:
        """The total number of sites."""
        return self.num_bath + 1

    @property
    def beta(self) -> float:
        """float : Inverse temperature .math:`1/T`"""
        return 1 / self.temp

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
            raise ValueError(
                f"Dimension of the new bath energy {eps_bath.shape} "
                f"does not match number of baths {self.num_bath}"
            )
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
            raise ValueError(
                f"Dimension of the new hybridization {v.shape} "
                f"does not match number of baths {self.num_bath}"
            )
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

    def _hamiltonian_data(self, up_states, dn_states):
        """Gets called by the `hamilton_operator`-method of the abstract base class."""
        u = np.append(self.u, np.zeros(self.num_bath))
        eps = np.append(self.eps_imp - self.mu, self.eps_bath)
        hopping = lambda i, j: self.v[j - 1] if i == 0 else 0  # noqa

        yield from project_onsite_energy(up_states, dn_states, eps)
        yield from project_hubbard_inter(up_states, dn_states, u)
        for j in range(self.num_bath):
            yield from project_hopping(up_states, dn_states, 0, j + 1, self.v[j])

    def impurity_gf0(self, z):
        return 1 / (z + self.mu + self.eps_imp - self.hybridization_func(z))

    def impurity_gf(self, z, sigma=UP):
        return greens_function_lehmann(
            self, z, beta=1 / self.temp, pos=0, sigma=sigma
        ).gf

    def pformat(self):
        return (
            f"U={self.u}, ε_i={self.eps_imp}, ε_b={self.eps_bath}, v={self.v}, "
            f"μ={self.mu}, T={self.temp}"
        )
