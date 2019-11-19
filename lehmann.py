# -*- coding: utf-8 -*-
"""
Created on 17 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
import scipy.linalg as la
from scitools import Plot, Matrix
from cmpy import FBasis, HamiltonOperator, Hamiltonian


def hamilton_operator(operators):
    (c1u, c2u), (c1d, c2d) = operators
    u_op = (c1u.dag * c1u * c1d.dag * c1d) + (c2u.dag * c2u * c2d.dag * c2d)
    eps_op = (c1u.dag * c1u) + (c2u.dag * c2u) + (c1d.dag * c1d) + (c2d.dag * c2d)
    t_op = (c1u.dag * c2u + c2u.dag * c1u) + (c1d.dag * c2d + c2d.dag * c1d)
    return HamiltonOperator(u=u_op, eps=eps_op, t=t_op)


def main():
    u, eps, t = 4, 2, 1

    basis = FBasis(2)
    basis.sort()

    ops = basis.build_annihilation_ops()
    ham_op = hamilton_operator(ops)
    ham = Hamiltonian(ham_op.build(u=u, eps=eps, t=t).todense())
    print(ham)
    ham.show(labels=basis.labels)


if __name__ == "__main__":
    main()
