# -*- coding: utf-8 -*-
"""
Created on 13 Jul 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from cmpy.core import Lattice, LatticePlot2D, TbHamiltonian


class IsingModel:

    DEFAULT = 1

    def __init__(self, vectors, j=1., h=0.):
        self.lattice = Lattice(vectors)
        self.spins = None
        self.j = j
        self.h = h

    @classmethod
    def square(cls, shape=(2, 1), j=1., h=0., name="A", a=1.):
        self = cls(a * np.eye(2), j, h)
        self.add_atom(name)
        if shape is not None:
            self.build(shape)
        return self

    def add_atom(self, name="A", pos=None):
        """ Add site to lattice cell and store it's energy.

        Parameters
        ----------
        name: str, optional
            Name of the site. The defualt is the origin of the cell
        pos: array_like, optional
            position of site in the lattice cell, default: "A"
        """
        # Add site to lattice
        self.lattice.add_atom(name, pos)

    def build(self, shape):
        self.lattice.calculate_distances(1)
        self.lattice.build(shape)
        self.spins = np.ones(self.lattice.n) * self.DEFAULT

    def set_spins(self, array):
        self.spins = np.asarray(array)

    def set_random_spins(self):
        self.spins = np.random.choice((-1, 1), size=self.lattice.n)

    def set_spin(self, n, value):
        i = self.lattice.get_list_idx(n, 0)
        self.spins[i] = value

    def flip_spin(self, n):
        i = self.lattice.get_list_idx(n, 0)
        self.spins[i] *= -1

    def hamiltonian(self):
        n = self.lattice.n
        ham = TbHamiltonian.zeros(n)
        for i in range(n):
            ham.set_energy(i, -self.h*self.spins[i])
            for distidx, j in self.lattice.iter_neighbor_indices(i):
                if j > i:
                    value = -self.j * self.spins[i] * self.spins[j]
                    ham[i, j] = ham[j, i] = value
        return ham

    def show(self):
        positions = np.asarray([self.lattice.position(i) for i in range(self.lattice.n)])
        spin_1 = positions[self.spins == self.DEFAULT]
        spin_2 = positions[self.spins != self.DEFAULT]
        segments = list()
        for i in range(self.lattice.n):
            neighbours = self.lattice.neighbours[i]
            for i_hop in range(len(self.lattice.distances)):
                for j in neighbours[i_hop]:
                    if j > i:
                        segments.append([positions[i], positions[j]])

        plot = LatticePlot2D()
        if len(spin_1):
            plot.draw_sites(np.asarray(spin_1), col="k")
        if len(spin_2):
            plot.draw_sites(np.asarray(spin_2), col="r")
        plot.draw_hoppings(segments)
        plot.show()
