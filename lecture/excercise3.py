# coding: utf-8
#
# This code is part of cmpy.
# 
# Copyright (c) 2021, Dylan Jones, Nico Unglert

import numpy as np
from scipy import sparse
from bisect import bisect_left
import matplotlib.pyplot as plt
from typing import Iterable, Callable, Optional, Sequence
from cmpy.models import ModelParameters
from cmpy.basis import Basis, overlap, occupations, binstr
from cmpy.matrix import Matrix


def project_up(up_idx: (int, np.ndarray), num_dn_states: int,
               dn_indices: np.ndarray) -> np.ndarray:
    """Projects spin-up states onto the full basis.

    Parameters
    ----------
    up_idx: int or ndarray
        The index/indices for the projection.
    num_dn_states: int
        The total number of spin-down states of the basis(-sector).
    dn_indices: ndarray
        An array of the indices of all spin-down states in the basis(-sector).
    """
    return up_idx * num_dn_states + dn_indices


def project_dn(dn_idx: (int, np.ndarray), num_dn_states: int,
               up_indices: np.ndarray) -> np.ndarray:
    """Projects spin-down states onto the full basis.

    Parameters
    ----------
    dn_idx: int or ndarray
        The index/indices for the projection.
    num_dn_states: int
        The total number of spin-down states of the basis(-sector).
    up_indices: ndarray
        An array of the indices of all spin-up states in the basis(-sector).
    """
    return up_indices * num_dn_states + dn_idx


def project_elements_up(num_dn_states, up_idx, dn_indices, value, target=None):
    if not value:
        return

    origins = project_up(up_idx, num_dn_states, dn_indices)
    if target is None:
        targets = origins
    else:
        targets = project_up(target, num_dn_states, dn_indices)

    if isinstance(origins, int):
        yield origins, targets, value
    else:
        for row, col in zip(origins, targets):
            yield row, col, value


def project_elements_dn(num_dn_states, dn_idx, up_indices, value, target=None):
    if not value:
        return

    origins = project_dn(dn_idx, num_dn_states, up_indices)
    if target is None:
        targets = origins
    else:
        targets = project_dn(target, num_dn_states, up_indices)

    if isinstance(origins, int):
        yield origins, targets, value
    else:
        for row, col in zip(origins, targets):
            yield row, col, value


def project_onsite_energy(up_states, dn_states, eps):
    num_dn = len(dn_states)
    all_up, all_dn = np.arange(len(up_states)), np.arange(num_dn)

    for up_idx, up in enumerate(up_states):
        weights = occupations(up)
        energy = np.sum(eps[:weights.size] * weights)
        yield from project_elements_up(num_dn, up_idx, all_dn, energy)

    for dn_idx, dn in enumerate(dn_states):
        weights = occupations(dn)
        energy = np.sum(eps[:weights.size] * weights)
        yield from project_elements_dn(num_dn, dn_idx, all_up, energy)


def project_interaction(up_states, dn_states, u):
    num_dn = len(dn_states)
    for up_idx, up in enumerate(up_states):
        for dn_idx, dn in enumerate(dn_states):
            weights = overlap(up, dn)
            interaction = np.sum(u[:weights.size] * weights)
            yield from project_elements_up(num_dn, up_idx, dn_idx, interaction)


def _hopping_candidates(num_sites, state, pos):
    results = []
    op = 1 << pos
    occ = state & op

    tmp = state ^ op  # Annihilate or create electron at `pos`
    for pos2 in range(num_sites):
        if pos >= pos2:
            continue
        op2 = (1 << pos2)
        occ2 = state & op2
        # Hopping from `pos` to `pos2` possible
        if occ and not occ2:
            new = tmp ^ op2
            results.append((pos2, new))
        # Hopping from `pos2` to `pos` possible
        elif not occ and occ2:
            new = tmp ^ op2
            results.append((pos2, new))

    return results


def _ordering_phase(state, pos1, pos2=0):
    if pos1 == pos2:
        return 0
    i0, i1 = sorted([pos1, pos2])
    particles = binstr(state)[i0 + 1:i1].count("1")
    return +1 if particles % 2 == 0 else -1


def _compute_hopping(num_sites, states, pos, hopping):
    for i, state in enumerate(states):
        for pos2, new in _hopping_candidates(num_sites, state, pos):
            try:
                t = hopping(pos, pos2)
            except TypeError:
                t = hopping
            if t:
                j = bisect_left(states, new)
                sign = _ordering_phase(state, pos, pos2)
                value = sign * t
                yield i, j, value


def project_site_hopping(up_states, dn_states, num_sites: int,
                         hopping: (Callable, Iterable, float), pos: int):
    num_dn = len(dn_states)
    all_up, all_dn = np.arange(len(up_states)), np.arange(num_dn)

    for up_idx, target, amp in _compute_hopping(num_sites, up_states, pos, hopping):
        yield from project_elements_up(num_dn, up_idx, all_dn, amp, target=target)

    for dn_idx, target, amp in _compute_hopping(num_sites, dn_states, pos, hopping):
        yield from project_elements_dn(num_dn, dn_idx, all_up, amp, target=target)


def project_hopping(up_states, dn_states, num_sites, hopping: (Callable, Iterable, float)):
    for pos in range(num_sites):
        yield from project_site_hopping(up_states, dn_states, num_sites, hopping, pos)


def siam_hamiltonian_data(up_states, dn_states, u, eps_imp, eps_bath, v):
    eps_bath = np.atleast_1d(eps_bath)
    v = np.atleast_1d(v)

    num_bath = len(eps_bath)
    num_sites = num_bath + 1
    u = np.append(u, np.zeros(num_bath))
    eps = np.append(eps_imp, eps_bath)
    hop = lambda i, j: v[j - 1] if i == 0 else 0  # noqa

    yield from project_onsite_energy(up_states, dn_states, eps)
    yield from project_interaction(up_states, dn_states, u)
    yield from project_site_hopping(up_states, dn_states, num_sites, hop, pos=0)


def siam_hamiltonian(up_states, dn_states, u, eps_imp, eps_bath, v):
    rows, cols, data = list(), list(), list()
    for row, col, value in siam_hamiltonian_data(up_states, dn_states, u, eps_imp, eps_bath, v):
        rows.append(row)
        cols.append(col)
        data.append(value)

    size = len(up_states) * len(dn_states)
    ham = sparse.csr_matrix((data, (rows, cols)), shape=(size, size))
    return ham


class SIAM(ModelParameters):

    def __init__(self, u: (float, Sequence[float]) = 2.0,
                 eps_imp: (float, Sequence[float]) = 0.0,
                 eps_bath: (float, Sequence[float]) = 0.0,
                 v: (float, Sequence[float]) = 1.0,
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
        super().__init__(u=u, eps_imp=eps_imp, eps_bath=eps_bath, v=v, mu=mu, temp=temp)
        self.basis = Basis(num_sites)

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

    def update_bath_energy(self, eps_bath: (float, np.ndarray)) -> None:
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

    def update_hybridization(self, v: (float, np.ndarray)) -> None:
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

    def pformat(self):
        return f"U={self.u}, ε_i={self.eps_imp}, ε_b={self.eps_bath}, v={self.v}, " \
               f"μ={self.mu}, T={self.temp}"

    def hamiltonian(self, n_up=None, n_dn=None):
        sector = self.basis.get_sector(n_up, n_dn)
        up, dn = sector.up_states, sector.dn_states
        return siam_hamiltonian(up, dn, self.u, self.eps_imp, self.eps_bath, self.v)

    def iter_fillings(self):
        for n_dn in range(self.num_sites + 1):
            for n_up in range(self.num_sites + 1):
                yield n_up, n_dn


def main():
    num_bath = 1
    u = 2
    eps_imp = 0
    eps_bath = 0 * np.ones(num_bath)
    v = 1 * np.ones(num_bath)

    siam = SIAM(u, eps_imp, eps_bath, v, mu=0)
    print(siam)

    # Hamiltonian of full Fock-basis
    #ham = siam.hamiltonian()
    #ham = Matrix(ham.toarray())
    #sector = siam.basis.get_sector()
    # ham.show(show=False, ticklabels=sector.state_labels(), values=True)
    # plt.show()

    # Compute groundstate explicitly
    #eigvals, eigvecs = ham.eigh()
    #i0 = np.argmin(eigvals)
    #e_gs = eigvals[i0]
    #gs = eigvecs[:, i0]

    #print(f"Ground state (E={e_gs:.2f}):")
    # print(gs)
    print()

    # Compute groundstate by sectors
    gs = None
    e_gs = np.infty
    gs_sec = None
    for n_up, n_dn in siam.iter_fillings():
        print(f"Sector [{n_up}, {n_dn}]")
        ham = siam.hamiltonian(n_up, n_dn)
        eigvals, eigvecs = np.linalg.eigh(ham.toarray())
        i0 = np.argmin(eigvals)
        e0 = eigvals[i0]
        if e0 < e_gs:
            e_gs = e0
            gs = eigvecs[:, i0]
            gs_sec = [n_up, n_dn]
    print(f"Ground state (E={e_gs:.2f}, sector {gs_sec}):")
    print(gs)


if __name__ == "__main__":
    main()
