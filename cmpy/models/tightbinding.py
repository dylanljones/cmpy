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
from lattpy import Lattice, DispersionPath
from cmpy.matrix import Matrix
from .abc import AbstractModel
from typing import Optional, Union, Any, Sequence, TypeVar, NewType


class BaseTightBindingModel(AbstractModel):

    def __init__(self, eps: Union[float, Sequence[float]] = 0.0,
                 t: Union[float, Sequence[float]] = 1.0,
                 mu: Optional[float] = 0.0,
                 temp: Optional[float] = 0.0):
        """Initializes the ``TightBindingModel``.

        Parameters
        ----------
        eps: float or Sequence, optional
            The onsite energy of the tight-binding model. The default value is ``0``.
        t: float or Sequence, optional
            The hopping parameter of the tight-binding model. The default value is ``1``.
        mu: float, optional
            Optional chemical potential. The default is ``0``.
        temp: float, optional
            Optional temperature in kelvin. The default is ``0``.
        """
        super().__init__(eps=eps, t=t, mu=mu, temp=temp)

    def pformat(self):
        return f"ε={self.eps}, t={self.t}, μ={self.mu}, T={self.temp}"


def get_bandpath(points, names=None, cycle=True):
    if isinstance(points, DispersionPath):
        path = points
    else:
        path = DispersionPath(len(points[0]))
        path.add_points(points, names)
        if cycle:
            path.cycle()
    return path


class TightBinding(Lattice):

    def __init__(self, vectors, atom=None, pos=None, energy=0., hopping=1., **atom_kwargs):
        super().__init__(vectors, atom, pos, energy=energy, **atom_kwargs)
        self.energies = None
        self.hoppings = list()
        if self.num_base:
            self.set_hopping(*np.atleast_1d(hopping))

    def copy(self):
        new = super().copy()
        new.hoppings = self.hoppings.copy()
        new.energies = self.energies.copy()
        return new

    def add_atom(self, energy=0.0, pos=None, atom=None, neighbours=0, **kwargs):
        return super().add_atom(pos, atom, neighbours, energy=energy)

    def set_hopping(self, *t):
        self.hoppings = list(t)
        if self.num_dist != len(t):
            self.calculate_distances(len(t))

    def build(self, shape, inbound=True, periodic=None, pos=None, window=None):
        super().build(shape, inbound, periodic, pos, window)
        self.energies = np.zeros(self.num_sites)
        for i in range(self.num_sites):
            self.energies[i] = self.get_energy(self.data.indices[i, -1])

    def get_energy(self, alpha=0):
        atom = self.atoms[alpha]
        return atom.kwargs.get("energy", 0.)

    def energy(self, i):
        idx = self.data.indices[i]
        atom = self.atoms[idx[-1]]
        return atom.kwargs.get("energy", 0.)

    def get_hopping(self, distidx=0):
        return self.hoppings[distidx]

    def cell_hamiltonian(self, dtype=None):
        n = self.num_base
        ham = Matrix.zeros(n, dtype=dtype)
        for alpha in range(self.num_base):
            ham[alpha, alpha] = self.get_energy(alpha)
            for distidx in range(self.num_dist):
                t = self.get_hopping(distidx)
                for idx in self.get_neighbours(alpha=alpha, distidx=distidx):
                    alpha2 = idx[-1]
                    if alpha2 != alpha:
                        try:
                            ham[alpha, alpha2] = t
                        except IndexError:
                            pass
        return ham

    def transform_cell_hamiltonian(self, k, cell_ham=None):
        if cell_ham is None:
            ham = self.cell_hamiltonian(np.complex)
        else:
            ham = cell_ham.astype(dtype=np.complex)
        # One atom in the unit cell
        if self.num_base == 1:
            for distidx in range(self.num_dist):
                t = self.get_hopping(distidx)
                nn_vecs = self.get_neighbour_vectors(alpha=0, distidx=distidx)
                ham += t * np.sum([np.exp(1j * np.dot(k, v)) for v in nn_vecs])
        # Multiple atoms in the unit cell
        else:
            for i in range(self.num_base):
                for j in range(self.num_base):
                    if i != j:
                        for distidx in range(self.num_dist):
                            nn_vecs = self.get_neighbour_vectors(alpha=j, distidx=distidx)
                            ham[i, j] = ham[i, j] * np.sum([np.exp(1j * np.dot(k, v)) for v in nn_vecs])
        return ham

    def hamiltonian(self):
        ham = Matrix.zeros(self.num_sites)
        for i in range(self.num_sites):
            ham[i, i] = self.energies[i]
            for distidx in range(self.num_dist):
                t = self.get_hopping(distidx)
                for j in self.neighbours(i, distidx, unique=True):
                    ham[i, j] = t
                    ham[j, i] = t
        return ham

    def dispersion(self, k):
        disp = np.zeros((len(k), self.num_base), dtype=np.complex64)
        cell_ham = self.cell_hamiltonian()
        for i, _k in enumerate(k):
            ham_k = self.transform_cell_hamiltonian(_k, cell_ham)
            disp[i] = ham_k.eigvals()
        return disp[0] if len(k) == 1 else disp

    def bands(self, points, names=None, n=1000, cycle=True):
        path = get_bandpath(points, names, cycle)
        k = path.build(n)
        return self.dispersion(k)
