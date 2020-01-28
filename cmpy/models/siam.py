# -*- coding: utf-8 -*-
"""
Created on 15 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from cmpy.core import Basis, Matrix, HamiltonOperator


def siam_hamilton_operator(operators):
    up, dn = operators
    c0u, up_ops = up[0], up[1:]
    c0d, dn_ops = dn[0], dn[1:]

    u_op = c0u.T * c0u * c0d.T * c0d
    eps_imp_op = c0u.T * c0u + c0d.T * c0d

    eps_list, v_list = list(), list()
    for ciu, cid in zip(up_ops, dn_ops):
        eps = ciu.T * ciu + cid.T * cid
        hop = (c0u.T * ciu + ciu.T * c0u) + (c0d.T * cid + cid.T * c0d)
        eps_list.append(eps)
        v_list.append(hop)

    return HamiltonOperator(u=u_op, eps_imp=eps_imp_op, eps_bath=eps_list, v=v_list)


class Siam:

    def __init__(self, u, eps_imp, eps_bath, v, mu=None, beta=0., n=None, s=None):
        """ Initilizes the single impurity Anderson model

        Parameters
        ----------
        u: float
        eps_imp: float
        eps_bath: float or array_like
        v: float or array_like
        mu: float, optional
        n: array_like or int, optional
        s: array_like or int, optional
        """
        eps_bath = [eps_bath] if not hasattr(eps_bath, "__len__") else eps_bath
        v = [v] if not hasattr(v, "__len__") else v

        self.n_sites = len(eps_bath) + 1

        self.u = float(u)
        self.eps_imp = float(eps_imp)
        self.eps_bath = np.asarray(eps_bath, dtype="float")
        self.v = np.asarray(v, dtype="float")
        self.mu = u / 2 if mu is None else mu
        self.beta = beta

        self.basis = None
        self.ops = None
        self.ham_op = None

        self.init_basis(n, s)

    def _update_operators(self):
        self.ops = self.basis.c_operators()
        self.ham_op = siam_hamilton_operator(self.ops)

    def _reset_operators(self):
        self.ops = None
        self.ham_op = None

    def set_sector(self, n=None, s=None):
        self.basis = Basis(self.n_sites, n=n, s=s)
        self._reset_operators()

    def sort_basis(self, key=None):
        self.basis.sort(key)
        self._reset_operators()

    def init_basis(self, n=None, s=None, key=None):
        self.basis = Basis(self.n_sites, n=n, s=s)
        self.sort_basis(key)
        self._reset_operators()

    @property
    def c_up(self):
        return self.ops[0][0]

    @property
    def c_down(self):
        return self.ops[1][0]

    @property
    def n_bath(self):
        return self.n_sites - 1

    @property
    def labels(self):
        return self.basis.labels

    def update_bath_energy(self, eps_bath):
        eps_bath = [eps_bath] if not hasattr(eps_bath, "__len__") else eps_bath
        if len(eps_bath) != self.n_bath:
            raise ValueError("Number of bath-parameters doesn't match existing ones")
        self.eps_bath = np.asarray(eps_bath, dtype="float")

    def update_hybridization(self, v):
        v = [v] if not hasattr(v, "__len__") else v
        if len(v) != self.n_bath:
            raise ValueError("Number of bath-parameters doesn't match existing ones")
        self.v = np.asarray(v, dtype="float")

    def update_bath(self, eps_bath, v):
        self.update_bath_energy(eps_bath)
        self.update_hybridization(v)

    # =========================================================================

    def hamiltonian(self):
        if self.ops is None:
            self._update_operators()
        ham = self.ham_op.build(u=self.u, eps_imp=self.eps_imp, eps_bath=self.eps_bath, v=self.v)
        return Matrix(ham.dense)

    def hybridization(self, z):
        delta = self.v[np.newaxis, :]**2 / (z + self.mu - self.eps_bath[np.newaxis, :])
        return delta.sum(axis=0)

    def eig(self):
        return np.linalg.eig(self.hamiltonian())

    def impurity_gf_free(self, z):
        return 1/(z + self.mu - self.eps_imp - self.hybridization(z))
