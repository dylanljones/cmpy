# -*- coding: utf-8 -*-
"""
Created on 15 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
import numpy.linalg as la
from pyplot import Plot
from cmpy.core import UP, AbstractBasisModel, GreensFunction


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

# Reference functions taken from M. Potthof:
# 'Two-site dynamical mean-field theory'
def impurity_gf_free_ref(z, eps0, eps1, v):
    e = (eps1 - eps0) / 2
    r = np.sqrt(e*e + v*v)
    term1 = (r - e) / (z - e - r)
    term2 = (r + e) / (z - e + r)
    return 1/(2*r) * (term1 + term2)


# Reference functions taken from E. Lange:
# 'Renormalized vs. unrenormalized perturbation-theoretical
# approaches to the Mott transition'
def impurity_gf_ref(z, u, v):
    sqrt16 = np.sqrt(u ** 2 + 16 * v ** 2)
    sqrt64 = np.sqrt(u ** 2 + 64 * v ** 2)
    a1 = 1/4 * (1 - (u ** 2 - 32 * v ** 2) / np.sqrt((u ** 2 + 64 * v ** 2) * (u ** 2 + 16 * v ** 2)))
    a2 = 1/2 - a1
    e1 = 1/4 * (sqrt64 - sqrt16)
    e2 = 1/4 * (sqrt64 + sqrt16)
    return (a1 / (z - e1) + a1 / (z + e1)) + (a2 / (z - e2) + a2 / (z + e2))

# =============================================================================


# noinspection PyAttributeOutsideInit
class Siam(AbstractBasisModel):

    def __init__(self, u, v=1, eps_imp=0., eps_bath=0, mu=None):
        eps = np.append(eps_imp, eps_bath).astype(np.float)
        v = np.atleast_1d(v).astype(np.float)
        u = np.append(u, np.zeros_like(eps_bath)).astype(np.float)
        mu = u[0] / 2 if mu is None else mu

        num_sites = len(eps)
        assert v.shape[-1] == num_sites - 1
        assert u.shape[-1] == num_sites

        super().__init__(num_sites, u=u, v=v, eps=eps, mu=mu)

    @classmethod
    def uniform(cls, n_sites, u, v=1., eps_imp=0., eps_bath=0., mu=None):
        ones = np.ones(n_sites - 1, dtype=np.float)
        return cls(u, v=ones * v, eps_imp=eps_imp, eps_bath=ones * eps_bath, mu=mu)

    @classmethod
    def mapping(cls, model, num_sites, eps_bath=None, mu=None,
                u_key="u", v_key="t", eps_key="eps", default=0.):
        ones = np.ones(num_sites - 1)
        u = getattr(model, u_key, default)
        v = getattr(model, v_key, default)
        eps_imp = getattr(model, eps_key, default)
        eps_bath = model.mu if eps_bath is None else eps_bath
        mu = model.mu if mu is None else mu
        return cls(u, v=ones * v, eps_imp=eps_imp, eps_bath=ones * eps_bath, mu=mu)

    @property
    def eps_imp(self):
        return self.eps[0]

    @property
    def eps_bath(self):
        return self.eps[1:]

    def __str__(self):
        paramstr = f"U={self.u[0]}, ε₀={self.eps[0]}, ε={self.eps[1:]}, V={self.v}, μ={self.mu}"
        return f"{self.__class__.__name__}({paramstr})"

    def plot(self, show=True, labels=True, grid=False):
        r = 1.0
        plot = Plot()
        plot.set_equal_aspect()

        draw_star(plot.ax, self, r=r, labels=labels)
        if grid:
            plot.grid(below_axis=True)

        plot.set_limits(1.3 * r, 1.3 * r)
        plot.show(enabled=show)
        return plot

    def update_bath_energy(self, eps_bath):
        eps = np.append(self.eps_imp, eps_bath).astype(np.float)
        assert eps.shape[-1] == self.num_sites
        self.eps = eps

    def update_hybridization(self, v):
        v = np.atleast_1d(v).astype(np.float)
        assert v.shape[-1] == self.num_sites - 1
        self.v = v

    def update_bath(self, eps_bath, v):
        self.update_bath_energy(eps_bath)
        self.update_hybridization(v)

    def hybridization(self, z):
        return np.sum(abs(self.v) ** 2 / (z[..., np.newaxis] - self.eps_bath), axis=-1)

    def build_matvec(self, matvec, x, hamop):
        hopping = lambda i, j: self.v[j - 1] if i == 0 else 0  # noqa

        hamop.apply_onsite_energy(matvec, x, self.eps)
        hamop.apply_interaction(matvec, x, self.u)
        hamop.apply_site_hopping(matvec, x, hopping, pos=0)

    def impurity_gf_0(self, z):
        return 1/(z + self.mu + self.eps_imp - self.hybridization(z + self.mu))

    def impurity_gf(self, z, beta=1., sigma=UP):
        gf = GreensFunction(z + self.mu, beta, pos=0, sigma=sigma)

        # for n_up, n_dn in self.iter_sector_keys(2):
        for n_dn in self.sector_keys:
            for n_up in self.sector_keys:
                # Solve current particle sector
                sector = self.get_sector(n_up, n_dn)
                hamop = self.hamiltonian_op(sector=sector)
                ham = hamop.matrix(np.eye(*hamop.shape))
                eigvals, eigvecs = la.eigh(ham)

                # Update partition and Green's function
                gf.accumulate_partition(eigvals)
                if (n_up + 1) in self.sector_keys:
                    sector_p1 = self.get_sector(n_up + 1, n_dn)
                    hamop_p1 = self.hamiltonian_op(sector=sector_p1)
                    gf.accumulate(hamop_p1, sector, sector_p1, eigvals, eigvecs)

        return gf.y / gf.partition

    def impurity_gf_ref(self, z):
        return impurity_gf_ref(z, self.u[0], self.v[0])

    def impurity_spectral_0(self, z):
        return -self.impurity_gf_0(z).imag

    def impurity_spectral(self, z, beta=1., sigma=UP):
        return -self.impurity_gf(z, beta, sigma).imag

    def impurity_spectral_ref(self, z):
        return -self.impurity_gf_ref(z).imag
