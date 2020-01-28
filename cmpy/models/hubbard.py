# -*- coding: utf-8 -*-
"""
Created on 15 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from cmpy.core import Basis


class HubbardModel:

    def __init__(self, u, t, eps=0, mu=None, beta=0., n=None, s=None):
        """ Initilizes the single impurity Anderson model

        Parameters
        ----------
        u: float
        t: float or array_like
        eps float or array_like, optional
        mu: float, optional
        n: array_like or int, optional
        s: array_like or int, optional
        """
        eps = [eps] if not hasattr(eps, "__len__") else eps
        v = [t] if not hasattr(t, "__len__") else t

        self.n_sites = len(eps)

        self.u = float(u)
        self.eps = np.asarray(eps, dtype="float")
        self.t = np.asarray(v, dtype="float")
        self.mu = u / 2 if mu is None else mu
        self.beta = beta

        self.basis = Basis(self.n_sites, n=n, s=s)
        self.sort_basis()

    def set_sector(self, n=None, s=None):
        self.basis = Basis(self.n_sites, n=n, s=s)

    def sort_basis(self, key=None):
        self.basis.sort(key)

    @property
    def labels(self):
        return self.basis.labels
