# coding: utf-8
"""
Created on 25 May 2020
Author: Dylan Jones
"""
import numpy as np


class HsField:

    """ Configuration class representing the discrete Hubbard-Stratonovich (HS) field."""

    def __init__(self, n_sites, time_steps=0, array=None, dtype=np.int8):
        self.dtype = dtype
        self.n_sites = n_sites
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
        return np.reshape(self.config, shape, order)

    def __getitem__(self, item):
        return self._config[item]

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