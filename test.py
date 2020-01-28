# -*- coding: utf-8 -*-
"""
Created on 11 Nov 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import time
import itertools
import numpy as np
import cmpy as cp
from cmpy import Basis, Lattice, Matrix
import colorcet as cc
# from cmpy.models.tightbinding import TightBindingModel


class LatticeModel:

    def __init__(self, vectors=np.eye(2), lattice=None, *args, **kwargs):
        self.lattice = Lattice(vectors) if lattice is None else lattice
        self.basis = None

    @property
    def n_sites(self):
        return self.lattice.n_sites

    def add_atom(self, name, pos=None, neighbour_dist=0):
        self.lattice.add_atom(name, pos, neighbour_dist)

    def calculate_distances(self):
        self.lattice.calculate_distances(1)

    def build(self, shape, cycling=True, **kwargs):
        self.lattice.build(shape)
        if cycling:
            self.lattice.set_periodic_boundary()
        self.basis = Basis(self.lattice.n_sites, **kwargs)

    def set_sector(self, n=None, s=None):
        self.basis = Basis(self.n_sites, n=n, s=s)

    def sort_basis(self, key=None):
        self.basis.sort(key)


class HubbardModel(LatticeModel):

    def __init__(self, vectors=np.eye(2), u=4, t=1, eps=0, mu=0, lattice=None):
        super().__init__(vectors, lattice)
        self.u = u
        self.t = t
        self.eps = eps
        self.mu = mu if mu is not None else u/2  # Half filling if None

    @classmethod
    def square(cls, u=4, t=1, eps=0, mu=0, atom_name="A", a=1.):
        self = cls(np.eye(2) * a, u, t, eps, mu)
        self.add_atom(name=atom_name)
        self.calculate_distances()
        return self

    def __str__(self):
        return f"HubbardModel(u={self.u}, t={self.t}, mu={self.mu})"

    def ham_kinetic(self):
        energy = self.u/2 - self.mu
        ham = energy * np.eye(self.lattice.n_sites, dtype=np.float64)
        for i in range(self.lattice.n_sites):
            for j in self.lattice.nearest_neighbours(i):
                if i > j:
                    ham[i, j] = -self.t
                    ham[j, i] = -np.conj(self.t)
        return ham

    def build(self, shape, cycling=True, n=None, s=None):
        super().build(shape, cycling, n=n, s=s)

    def hamiltonian(self):
        n_states = self.basis.n
        ham = Matrix.zeros(n_states)
        for i in range(n_states):
            st1 = self.basis[i]
            eps = st1.n * self.eps
            u = bin(st1.up & st1.dn).count("1") * self.u
            ham[i, i] = eps + u - self.mu
            for j in range(i + 1, n_states):
                st2 = self.basis[j]
                indices = st1.hopping_indices(st2)
                if indices is not None:
                    idx0, idx1 = indices
                    if idx0 in self.lattice.nearest_neighbours(idx1):
                        ham[i, j] = -self.t
                        ham[j, i] = -np.conj(self.t)
        return ham

    def show_hamiltonian(self, show=True, cmap=None):
        ham = self.hamiltonian()
        show_info = ham.shape[0] <= 40
        labels = self.basis.labels if show_info else None
        return ham.show(show, cmap=cmap, values=show_info, y_ticklabels=labels, x_ticklabels=labels, norm_offset=0.0)


def main():
    model = HubbardModel.square(u=4, t=1)

    model.build((2, 1))
    model.basis.sort()
    model.show_hamiltonian()


if __name__ == "__main__":
    main()
