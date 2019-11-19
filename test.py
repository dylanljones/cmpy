# -*- coding: utf-8 -*-
"""
Created on 11 Nov 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
import numpy.linalg as la
from cmpy.models.siam import Siam
from cmpy.core.basis import FBasis
from cmpy.core.operators import HamiltonOperator
from scitools import Matrix

from tqdm import tqdm


def hamiltonian(up, dn):
    eps = sum([up[ii].T * up[ii] + dn[ii].T * dn[ii] for ii in range(len(up))])
    t_ops = list()
    for ii in tqdm(list(range(len(dn) - 1))):
        t_ops.append(up[ii].dag*up[ii + 1] + up[ii + 1].dag*up[ii] +
                     dn[ii].dag*dn[ii + 1] + dn[ii + 1].dag*dn[ii])
    hamop = HamiltonOperator(eps=eps, t=t_ops)
    return hamop


def hamiltonian2(up, dn):
    (c1u, c2u), (c1d, c2d) = up, dn
    u_op = (c1u.dag * c1u * c1d.dag * c1d) + (c2u.dag * c2u * c2d.dag * c2d)
    eps_op = (c1u.dag * c1u) + (c2u.dag * c2u) + (c1d.dag * c1d) + (c2d.dag * c2d)
    t_op = (c1u.dag * c2u + c2u.dag * c1u) + (c1d.dag * c2d + c2d.dag * c1d)
    return HamiltonOperator(u=u_op, eps=eps_op, t=t_op)



def main():
    basis = FBasis(n_sites=2)
    basis.sort()
    up, dn = basis.build_annihilation_ops()
    hamop = hamiltonian(up, dn)
    ham = Matrix(hamop.build(eps=1, t=[1, 1]).todense())
    # print(ham)
    # ham.show()
    print(la.eig(ham))

    #siam = Siam(u=4, eps_imp=0, eps_bath=2, v=1)
    #siam.init_basis(n=[1, 2], key=lambda s: s.n)
    #ham = siam.hamiltonian()
    #ham.show(labels=siam.labels)



if __name__ == "__main__":
    main()
