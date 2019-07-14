# -*- coding: utf-8 -*-
"""
Created on 26 Mar 2019
author: Dylan

project: cmpy2
version: 1.0
"""
from itertools import product
from sciutils import *
from cmpy import *
from cmpy.hubbard import *


def main():
    model = Siam(eps=2, v=5, mu=0)
    ham = model.hamiltonian()
    ham.show()


if __name__ == "__main__":
    main()

