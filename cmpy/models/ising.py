# coding: utf-8
"""
Created on 27 May 2020
Author: Dylan Jones
"""
import random
import numpy as np
from lattpy import Lattice
from pyplot import Plot
from cmpy.qmc import HsField


class IsingModel(Lattice):

    def __init__(self, j=1., h=0., temp=0.):
        super().__init__(vectors=np.eye(2))
        self.add_atom()
        self.calculate_distances(1)
        self.field = None
        self.j = j
        self.h = h
        self.temp = temp

    def build(self, shape, inbound=True, pos=None):
        super().build(shape, inbound, pos)
        self.field = HsField(self.n_sites)

    def get_energy_element(self, i, unique=False):
        s = self.field.config[i]
        s_neighbours = [self.field.config[j] for j in self.nearest_neighbours(i, unique=unique)]
        return - self.j * s * np.sum(s_neighbours) - s * self.h

    def try_flip(self, i):
        # Check energy difference from potential spin flip
        delta_e = - self.get_energy_element(i)
        # Update field if energy advantage
        if delta_e < 0:
            self.field.update(i)
        # At finite temp > 0: Metropolis acceptance
        elif self.temp > 0 and random.random() < np.exp(-1/self.temp * delta_e):
            self.field.update(i)

    def energy(self):
        energy = 0.0
        for i in range(self.n_sites):
            energy += self.get_energy_element(i, unique=True)
        return energy / self.n_sites

    def magnetization(self):
        return self.field.mean()

    def info_string(self):
        return f"<M>: {self.magnetization():.2f}, <E>: {self.energy():.2f}"

    def simulate(self, n_check=10000):
        shape = int(self.shape[0]) + 1, int(self.shape[1]) + 1
        plot = Plot()
        im = plot.ax.imshow(self.field.reshape(shape).T, cmap="RdBu", vmin=-1.2, vmax=1.3)
        plot.invert_yaxis()
        plot.set_equal_aspect()
        indices = np.arange(self.n_sites)
        i = 0
        while True:
            np.random.shuffle(indices)
            for idx in indices:
                #  idx = i % self.n_sites  # random.randint(0, self.n_sites-1)
                self.try_flip(idx)
                if i % n_check == 0:
                    im.set_data(self.field.reshape(shape).T)
                    plot.redraw()
                    print("\r" + self.info_string(), end="", flush=True)
                i += 1
            if len(np.unique(self.field.config)) == 1:
                break
        plot.show()
