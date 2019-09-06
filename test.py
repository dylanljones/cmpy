# -*- coding: utf-8 -*-
"""
Created on 26 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
from itertools import product
from sciutils import eta, Plot, prange
from cmpy import *
from cmpy.tightbinding import *
from cmpy.hubbard import *

# =========================================================================


def main():
    model = TbDevice.square_basis((20, 4), basis="p3", soc=1)
    model.sort_states()
    model.set_disorder(1)
    for _ in prange(100000):
        model.shuffle()
        model.transmission()



if __name__ == "__main__":
    main()
