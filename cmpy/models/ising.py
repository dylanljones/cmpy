# coding: utf-8
#
# This code is part of cmpy.
# 
# Copyright (c) 2021, Dylan Jones

import random
import logging
import numpy as np
import scipy.interpolate
from scipy.sparse import csr_matrix
from lattpy import Lattice, LatticeData
from typing import Optional, Union, Callable, Sequence, Iterator

logger = logging.getLogger(__name__)


class HsField:

    """ Configuration class representing the discrete Hubbard-Stratonovich (HS) field."""

    def __init__(self, num_sites, time_steps=0, array=None, dtype=np.int8):
        self.dtype = dtype
        self.n_sites = num_sites
        self.time_steps = time_steps
        self._config = np.ndarray
        if array is not None:
            self._config = np.asarray(array, dtype=self.dtype)
        else:
            self.initialize()

    @property
    def config(self):
        return self._config if self.time_steps else self._config[:, 0]

    def __eq__(self, other):
        return np.all(self._config == other._config)  # noqa

    def copy(self):
        """ Creates a (deep) copy of the 'Configuration' instance
        Returns
        -------
        config: Configuration
        """
        return HsField(self.n_sites, self.time_steps, array=self._config.copy())

    def initialize(self):
        """ Initializes the configuration with a random distribution of -1 and +1 """
        # Create an array of random 0 and 1 and scale array to -1 and 1
        time_steps = max(1, self.time_steps)
        field = 2 * np.random.randint(0, 2, size=(self.n_sites, time_steps)) - 1
        self._config = field.astype(self.dtype)

    def update(self, i, t=0):
        """ Update element of array by flipping its spin-value
        Parameters
        ----------
        i: int
            Site index.
        t: int, optional
            Time slice index.
        """
        self._config[i, t] *= -1

    def mean(self):
        """ float: Computes the Monte-Carlo sample mean """
        return np.mean(self._config)

    def var(self):
        """ float: Computes the Monte-Carlo sample variance """
        return np.var(self._config)

    def reshape(self, shape, order="C"):
        shape = np.asarray(shape).astype(np.int64) + 1
        return np.reshape(self.config, shape, order)

    def __getitem__(self, item):
        return self._config[item]

    def __setitem__(self, key, value):
        self._config[key] = value

    def string_header(self, delim=" "):
        return r"i\l  " + delim.join([f"{i:^3}" for i in range(self.time_steps)])

    def string_bulk(self, delim=" "):
        rows = list()
        for site in range(self.n_sites):
            row = delim.join([f"{x:^3}" for x in self._config[site, :]])
            rows.append(f"{site:<3} [{row}]")
        return "\n".join(rows)

    def __str__(self):
        delim = " "
        string = self.string_header(delim) + "\n"
        string += self.string_bulk(delim)
        return string


def check_spin_flip(de, temp):
    metro = np.exp(-de / temp)
    if not isinstance(de, np.ndarray):
        return (de < 0) or (temp > 0 and random.random() < metro)
    return (de < 0) | ((de > 0) & (np.random.rand(len(de)) < metro))


class GridInterpolation:

    def __init__(self, positions, padding=0.5, step=1.0, method="nearest"):
        # Set up a regular grid of interpolation points
        x, y = positions.T
        xi = np.arange(x.min() - padding, x.max() + padding, step)
        yi = np.arange(y.min() - padding, y.max() + padding, step)
        xi, yi = np.meshgrid(xi, yi)

        self._method = method
        self._positions = positions
        self._grid = xi, yi

    def interpolate(self, z):
        zi = scipy.interpolate.griddata(self._positions, z, self._grid, method=self._method)
        xi, yi = self._grid
        return xi, yi, zi

    def __call__(self, z):
        return self.interpolate(z)


class IsingModel(Lattice):

    def __init__(self, vectors, j1=0, j2=-1.0, temp=0.):
        super().__init__(vectors)
        self.j1 = j1
        self.j2 = j2
        self.temp = temp
        self._config: Optional[np.ndarray] = None

    @property
    def config(self):
        if self._config is None:
            self.init_config(-1)
        return self._config

    def init_config(self, value: int = -1):
        """Initializes the spin configuration with the given value."""
        self._config = np.full(self.num_sites, fill_value=value)

    def shuffle_config(self, p: Union[float, Sequence[float]] = None) -> None:
        """Initializes the configuration with a random distribution of `-1` and `+1`.

        Parameters
        ----------
        p : float or Sequence, optional
            Optional sample probabilities of the values `-1` and `+1`.
            The first probability corresponds to `-1`, the second to `+1`.
            If `p` is a `float` the values are sample with the probabilities `(p, 1-p)`,
            if no `p` is passed equal probabilities of `0.5` are used for both values.
        """
        if p is None:
            p = [0.5, 0.5]
        elif isinstance(p, float):
            p = [p, 1-p]
        spins = np.random.choice([-1, +1], p=p, size=self.num_sites)
        self._config = spins

    # =========================================================================

    def hamiltonian_data(self):
        dmap = self.data.map()
        indices = dmap.indices
        hop_mask = dmap.hopping(0)
        data = np.zeros(dmap.size)
        data[dmap.onsite()] = self.j1 * self.config[:]
        data[hop_mask] = self.j2 * np.prod(self.config[indices[:, hop_mask]], axis=0)
        return indices, data

    def hamiltonian(self):
        num_sites = self.num_sites
        indices, data = self.hamiltonian_data()
        return csr_matrix((data, indices), shape=(num_sites, num_sites))

    def site_energy(self, index: int):
        s = self._config[index]
        s_neighbours = [self._config[j] for j in self.nearest_neighbors(index)]
        return self.j1 * s + self.j2 * s * np.sum(s_neighbours)

    def site_energies(self):
        spins = self.config
        config_padded = np.append(self.config, 0)
        neighbors = self.data.neighbors
        spins_neighbors = config_padded[neighbors]
        return self.j1 * spins + self.j2 * spins * np.sum(spins_neighbors, axis=1)

    # =========================================================================

    def energy(self, per_site: bool = True):
        energy = sum(self.site_energies())
        if per_site:
            energy /= self.num_sites
        return energy

    def flip_spins(self, index: Union[int, Sequence[int]]):
        self._config[index] *= -1

    def check_spin_flip(self, index: int) -> bool:
        """Checks if a spin flip of a site results in a energy advantage."""
        delta_e = self.site_energy(index)
        return check_spin_flip(delta_e, self.temp)

    def try_spin_flip(self, index: int):
        if self.check_spin_flip(index):
            self.flip_spins(index)

    def flip_random(self, num: int = None, delta_e=None):
        if delta_e is None:
            delta_e = -2 * self.site_energies()

        indices = np.arange(self.num_sites)
        if num is None:
            np.random.shuffle(indices)
        else:
            num = min(num, self.num_sites)
            indices = np.random.choice(indices, size=num, repeat=False)

        for i in indices:
            de = delta_e[i]
            if check_spin_flip(de, self.temp):
                self.flip_spins(i)
                # If a spin is flipped the energy difference of the site needs to be flipped
                # and the energy differences of the neighbors need to be updated
                delta_e[i] = - de
                for j in self.nearest_neighbors(i):
                    try:
                        delta_e[j] = - self.site_energy(j)
                    except IndexError:
                        pass
        return delta_e
