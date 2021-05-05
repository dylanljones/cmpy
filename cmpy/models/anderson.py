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
from typing import Optional, Union, Sequence
from .abc import AbstractManyBodyModel


class SingleImpurityAndersonModel(AbstractManyBodyModel, ABC):

    def __init__(self, u: Union[float, Sequence[float]] = 2.0,
                 eps_imp: Union[float, Sequence[float]] = 0.0,
                 eps_bath: Union[float, Sequence[float]] = 0.0,
                 v: Union[float, Sequence[float]] = 1.0,
                 mu: Optional[float] = 0.0,
                 temp: Optional[float] = 0.0):
        r"""Initializes the single impurity Anderson model

        Parameters
        ----------
        u: float
            The on-site interaction strength.
        eps_imp: float, optional
            The on-site energy of the impurity site. The default is `0`.
        eps_bath: float or (N) float np.ndarray
            The on-site energy of the bath site(s). If a float is given the model
            is set to one bath site, otherwise the number of bath sites is given
            by the number of energy values passed.
            If the SIAM is set to half filling, the bath energy can be fixed at
            .math:`\epsilon_B \mu = u/2`. If `None`is given, one bath site at
            half filling will be set up.
            The default is `None` (half filling with one bath site).
        v: float or (N) float np.ndarray
            The hopping energy between the impurity site and the bath site(s).
            The number of hopping parameters must match the number of bath energies
            passed, i.e. the number of bath sites. The default is `1`.
        mu: float, optional
            The chemical potential of the system. If `None` the system is set
            to half filling, meaning a chemical potential of .math:`\mu = u/2`.
            The default is `None` (half filling).
        temp: float, optional
            Optional temperature in kelvin. The default is ``0``.
        """
        eps_bath = u / 2 if eps_bath is None else eps_bath
        mu = u / 2 if mu is None else mu
        eps_bath = np.atleast_1d(eps_bath).astype(np.float64)
        v = np.atleast_1d(v).astype(np.float64)
        num_sites = len(eps_bath) + 1
        super().__init__(num_sites, u=u, eps_imp=eps_imp, eps_bath=eps_bath, v=v, mu=mu, temp=temp)

    @classmethod
    def half_filled(cls, u, eps_imp, v, temp=0.0):
        """Initializes a single impurity Anderson model at half filling."""
        return cls(u=u, eps_imp=eps_imp, eps_bath=u/2, v=v, mu=u/2, temp=temp)

    @property
    def num_bath(self) -> int:
        """The number of bath sites."""
        return len(self.eps_bath)

    @property
    def num_sites(self) -> int:
        """The total number of sites."""
        return self.num_bath + 1

    def update_bath_energy(self, eps_bath: Union[float, np.ndarray]) -> None:
        """Updates the on-site energies `eps_bath` of the bath sites.

        Parameters
        ----------
        eps_bath: float or (N) np.ndarray
            The energy values of the bath sites. If only one bath site
            is used a float value can be passed.
        """
        eps_bath = np.atleast_1d(eps_bath).astype(np.float64)
        assert eps_bath.shape[0] == self.num_bath
        self.eps_bath = eps_bath  # noqa

    def update_hybridization(self, v: Union[float, np.ndarray]) -> None:
        """Updates the hopping parameters `v` between the impurity and bath sites.

        Parameters
        ----------
        v: float or (N) np.ndarray
            The hopping parameters between the impurity and bath sites.
            If only one bath site is used a float value can be passed.
        """
        v = np.atleast_1d(v).astype(np.float64)
        assert v.shape[0] == self.num_bath
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
        return np.sum(np.abs(self.v) ** 2 / (z[..., np.newaxis] - self.eps_bath), axis=-1)

    def pformat(self):
        return f"U={self.u}, ε_i={self.eps_imp}, ε_b={self.eps_bath}, v={self.v}, " \
               f"μ={self.mu}, T={self.temp}"


def polar(r, phi):
    return r * np.cos(phi), r * np.sin(phi)


def draw_star(ax, siam, r=1.0, size=8, imp_size=11, labels=True):
    eps_bath = siam.eps_bath
    v = siam.v
    n = len(eps_bath)
    phi0 = 0.0 if n <= 2 else np.pi / 2

    imp_label = rf"U={siam.u[0]:.1f}, $\epsilon$={siam.eps_imp:.1f}"
    ax.scatter([0], [0], color="k", s=imp_size**2, zorder=2, label=imp_label)
    for i in range(n):
        phi = i * 2 * np.pi / n
        pos = polar(r, phi + phi0)
        ax.plot([0, pos[0]], [0, pos[1]], color="k", zorder=1)
        ax.scatter(*pos, s=size**2, color="C0", zorder=2)

        if labels:
            eps_label = rf"$\epsilon$={eps_bath[i]:.1f}"
            x, y = polar(r*1.2, phi + phi0)
            ax.text(x=x, y=y, s=eps_label, va="center", ha="center")

            v_label = rf"V={v[i]:.1f}"
            x, y = polar(r * 0.5, phi + 0.1 * np.pi + phi0)
            ax.text(x=x, y=y, s=v_label, va="center", ha="center")

    if labels:
        ax.legend()


# ========================== REFERENCES =======================================

# Reference functions taken from E. Lange:
# 'Renormalized vs. unrenormalized perturbation-theoretical
# approaches to the Mott transition'
def impurity_gf_ref(z, u, v):
    sqrt16 = np.sqrt(u**2 + 16 * v**2)
    sqrt64 = np.sqrt(u**2 + 64 * v**2)
    a1 = 1/4 * (1 - (u**2 - 32 * v**2) / np.sqrt((u**2 + 64 * v**2) * (u**2 + 16 * v**2)))
    a2 = 1/2 - a1
    e1 = 1/4 * (sqrt64 - sqrt16)
    e2 = 1/4 * (sqrt64 + sqrt16)
    return (a1 / (z - e1) + a1 / (z + e1)) + (a2 / (z - e2) + a2 / (z + e2))
