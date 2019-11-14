# -*- coding: utf-8 -*-
"""
Created on 11 Nov 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from cmpy.models.siam import Siam


def main():
    siam = Siam(u=4, eps_imp=0, eps_bath=2, v=1)
    siam.init_basis(n=[1, 2], key=lambda s: s.n)
    ham = siam.hamiltonian()
    ham.show(labels=siam.labels)



if __name__ == "__main__":
    main()
