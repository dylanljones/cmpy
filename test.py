#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from lattpy import Lattice

def visualize_states():
    from cmpy import Basis

    basis = Basis(2)
    print("\nbasis: \n", basis)
    sec = basis.get_sector(n_up=2, n_dn=2)
    getstates = basis.get_states()
    print("\nget_states: \n", getstates)
    states = list(sec.states)

    print("\nsec.states:")
    for i in states:
        print(i)

    latt = Lattice([[1,0],[0,1]])

    state = states[0]
    # print(state)
    # state.create(0)

    # state = sec.up.create(0)
    # print(state)



if __name__ == "__main__":
    # A=1
    visualize_states()