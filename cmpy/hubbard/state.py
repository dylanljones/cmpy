# -*- coding: utf-8 -*-
"""
Created on 14 Jul 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np


class State:

    UP_CHAR = "\u2191"
    DN_CHAR = "\u2193"

    def __init__(self, up, down):
        self.arr = np.asarray((up, down))

    @property
    def n(self):
        return self.arr.shape[1]

    @property
    def spin(self):
        return 1/2 * (np.sum(self[0]) - np.sum(self[1]))

    @property
    def num(self):
        return int(np.sum(self.arr))

    @property
    def up(self):
        return self.arr[0]

    @property
    def dn(self):
        return self.arr[1]

    @property
    def single(self):
        return np.any(self.arr, axis=0) ^ np.all(self.arr, axis=0).astype("int")

    @property
    def double(self):
        return np.all(self.arr, axis=0).astype("int")

    def site(self, i):
        return self.arr[:, i]

    def n_onsite(self):
        return np.sum(self.single)

    def n_interaction(self):
        return np.sum(self.double)

    def difference(self, other):
        up_diff = np.where((self.up != other.up) > 0)[0]
        dn_diff = np.where((self.dn != other.dn) > 0)[0]
        return up_diff, dn_diff

    def check_hopping(self, s):
        if self.num != s.num:
            return None
        if self.spin != s.spin:
            return None
        up_diff = np.where((self.up != s.up) > 0)[0]
        dn_diff = np.where((self.dn != s.dn) > 0)[0]
        hops = int((len(up_diff) + len(dn_diff)) / 2)
        if hops != 1:
            return None
        for diff in (up_diff, dn_diff):
            if diff.shape[0] > 0:
                return diff

    def __eq__(self, other):
        return np.all(self.arr == other.arr)

    def __getitem__(self, item):
        return self.arr[item]

    def _get_sitechar(self, i, latex=False):
        if self[0, i] and self[1, i]:
            return "d"
        elif self[0, i] and not self[1, i]:
            return self.UP_CHAR if not latex else r"$\uparrow$"
        elif not self[0, i] and self[1, i]:
            return self.DN_CHAR if not latex else r"$\downarrow$"
        return "."

    @property
    def repr(self):
        return " ".join([self._get_sitechar(i) for i in range(self.n)])

    def latex_string(self):
        return " ".join([self._get_sitechar(i, latex=True) for i in range(self.n)])

    def __repr__(self):
        return f"State({self.repr})"

    def __str__(self):
        string = self.repr + f" (N_e={self.num}, S={self.spin})"
        return string
