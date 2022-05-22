# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2022, Dylan Jones

import numpy as np
from scipy.sparse import csr_matrix
from ..operators import project_hubbard_inter, project_onsite_energy, project_hopping
from .abc import AbstractManyBodyModel


def _ham_data(up_states, dn_states, num_sites, neighbors, inter, eps, hop):
    """Generates the indices and values of the Hubbard Hamiltonian matrix elements."""
    energy = np.full(num_sites, eps)
    interaction = np.full(num_sites, inter)

    yield from project_onsite_energy(up_states, dn_states, energy)
    yield from project_hubbard_inter(up_states, dn_states, interaction)
    for i, j in neighbors:
        if i < j:
            yield from project_hopping(up_states, dn_states, num_sites, i, j, hop)


def hubbard_hamiltonian(sector, neighbors, inter=0.0, eps=0.0, hop=1.0):
    up = sector.up_states
    dn = sector.dn_states
    num_sites = sector.num_sites
    rows, cols, data = list(), list(), list()
    for i, j, val in _ham_data(up, dn, num_sites, neighbors, inter, eps, hop):
        rows.append(i)
        cols.append(j)
        data.append(val)
    return csr_matrix((data, (rows, cols)))


class HubbardModel(AbstractManyBodyModel):
    """Model class for the Hubbard model.

    The Hamiltonian of the Hubbard model is defined as
    .. math::
        H = U Σ_i n_{i↑}n_{i↓} + Σ_{iσ} ε_i c^†_{iσ}c_{iσ} + t Σ_{i,j,σ} c^†_{iσ}c_{jσ}

    Attributes
    ----------
    inter : float or Sequence, optional
        The onsite interaction energy of the model. The default value is ``2``.
    eps : float or Sequence, optional
        The onsite energy of the model. The default value is ``0``.
    eps_bath : float or Sequence, optional
        The onsite energy of the model. The default value is ``0``.
    hop : float or Sequence, optional
        The hopping parameter of the model. The default value is ``1``.
    mu : float, optional
        The chemical potential. The default is ``0``.
    """

    def __init__(self, *args, inter=0.0, eps=0.0, hop=1.0, mu=0.0):
        """Initializes the ``HubbardModel``."""
        if len(args) == 1:
            # Argument is a lattice instance
            latt = args[0]
            num_sites = latt.num_sites
            neighbors = latt.neighbor_pairs(True)[0]
        else:
            # Aurgument is the number of sites and the neighbor data
            num_sites, neighbors = args

        super().__init__(num_sites, inter=inter, eps=eps, hop=hop, mu=mu)
        self.neighbors = neighbors

    def pformat(self):
        return f"U={self.inter}, ε={self.eps}, t={self.hop}, μ={self.mu}"

    def _hamiltonian_data(self, up_states, dn_states):
        inter = self.inter
        eps = self.eps - self.mu
        hop = self.hop
        num_sites = self.num_sites
        neighbors = self.neighbors
        return _ham_data(up_states, dn_states, num_sites, neighbors, inter, eps, hop)
