# -*- coding: utf-8 -*-
"""
Created on 13 Jul 2019
author: Dylan Jones

project: cmpy
version: 1.0

ISING MODEL
"""
from sciutils import *
import random


class IsingModel:

    def __init__(self, shape, j=1., h=0., temp=0.):
        self.j = j
        self.h = h
        self.temp = temp

        self.array = 2 * np.random.randint(2, size=shape) - 1
        self.plot, self.im, self.text = None, None, None

    @property
    def n(self):
        return int(np.prod(self.shape))

    @property
    def shape(self):
        return self.array.shape

    @property
    def magnetization(self):
        return 1/self.n * abs(np.sum(self.array))

    @property
    def inner_energy(self):
        return self.j * np.sum(self.array)

    def __setitem__(self, key, value):
        self.array[key] = value

    def set_temp(self, t):
        self.temp = t

    def index(self, i):
        n, m = self.shape
        col, row = divmod(i, n)
        return int(row), int(col)

    def flip_spin(self, *idx):
        if len(idx) == 1:
            idx = self.index(idx)
        self.array[idx] *= -1

    def neighbor_indices(self, i, j):
        n, m = self.shape
        neighbors = list()
        neighbors.append(((i - 1) % n, j))
        neighbors.append(((i + 1) % n, j))
        neighbors.append((i, (j - 1) % m))
        neighbors.append((i, (j + 1) % m))
        return neighbors

    def check_flip_energy(self, i, j):
        s = self.array[i, j]
        neighbors = self.neighbor_indices(i, j)
        return 2 * s * np.sum([self.array[idx] for idx in neighbors])

    def try_flip(self, idx):
        i, j = self.index(idx)
        delta_e = self.check_flip_energy(i, j)
        if delta_e < 0:
            self.flip_spin(i, j)
        # metropolis acceptance
        elif self.temp > 0 and random.random() < np.exp(-1/self.temp * delta_e):
            self.flip_spin(i, j)

    def simulate(self, n_check=10000):
        self.init_plot()
        i = 0
        while True:
            idx = random.randint(0, self.n-1)
            self.try_flip(idx)
            if i % n_check == 0:
                self.update_plot(i)
                if len(np.unique(self.array)) == 1:
                    break
            i += 1
        self.plot.show()

    def equilibrium(self, n=50000):
        for i in range(n):
            idx = random.randint(0, self.n-1)
            self.try_flip(idx)
            if len(np.unique(self.array)) == 1:
                break

    def init_plot(self):
        self.plot = Plot()
        self.plot.set_title(f"$T={self.temp}$" + r" $J/k_B$")
        self.plot.set_equal_aspect()
        self.im = self.plot.ax.imshow(self.array.T, cmap="RdBu", vmin=-1.2, vmax=1.3)
        x = self.shape[1] * 0.95
        y = self.shape[1] * 0.05
        self.text = self.plot.text((x, y), "0", ha="right")

    def update_plot(self, t, sleep=1e-10):
        self.im.set_data(self.array)
        self.text.set_text(f"t={t:.0f}, M={self.magnetization:.2f}")
        self.plot.draw(sleep)

    def show(self):
        if self.plot is None:
            self.init_plot()
        self.plot.show()

    def wolf_cluster(self):
        idx = self.index(random.randint(0, self.n-1))
        p = 1 - np.exp(-2 / self.temp) if self.temp > 0 else 1

        pocket, cluster = [idx], [idx]
        while len(pocket) > 0:
            idx1 = random.choice(pocket)
            for idx2 in self.neighbor_indices(*idx1):
                if self.array[idx1] == self.array[idx2] and idx2 not in cluster:
                    if random.uniform(0., 1.) < p:
                        pocket.append(idx2)
                        cluster.append(idx2)
            pocket.remove(idx1)
        for idx in cluster:
            self.flip_spin(*idx)

    def cluster_equilibrium(self, n=100):
        for i in range(n):
            self.wolf_cluster()
            if len(np.unique(self.array)) == 1:
                break

    def cluster_mean_magnetization(self, n=10):
        m = np.zeros(n)
        for i in range(n):
            self.wolf_cluster()
            m[i] = self.magnetization
        return np.mean(m)

    def simulate_cluster(self):
        self.init_plot()
        i = 0
        while True:
            self.wolf_cluster()
            self.update_plot(i)
            if len(np.unique(self.array)) == 1:
                break
            i += 1
        self.plot.show()
