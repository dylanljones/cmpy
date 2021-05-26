#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from lattpy import Lattice
from cmpy.models import SingleImpurityAndersonModel

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



def mpl():
    import matplotlib.pyplot as plt
    # import matplotlib as mpl
    # mpl.rcParams.update(mpl.rcParamsDefault)
    import matplotlib.font_manager
    x = np.linspace(0,1,100)
    y = x
    plt.plot(x,y)
    # plt.rcParams['font.family'] = 'cursive'
    plt.xlabel("asdads A H B ", fontname="Times New Roman")
    plt.ylabel("asdads [eV]")

    # plt.rcParams['font.sans-serif'] = ['Tahoma',
    #                            'Lucida Grande', 'Verdana']
    plt.show()


def siam():
    import matplotlib.pyplot as plt
    siam = SingleImpurityAndersonModel(u=1.0, eps_imp=1.0, eps_bath=[2.0,2.0], v=[1.0,1.0])
    zz = np.linspace(-2,2,50) + 0.001j
    # siam.impurity_gf(zz, 10)
    gs = siam.get_total_gs()
    tt, overlaps = siam.t_evo_gr(gs, 0, 100, 10000)
    plt.plot(tt, overlaps)
    plt.show()

def test_project_elements():
    up_idx = 2
    dn_indices = np.array([0,1,2])
    num_dn_states = len(dn_indices)
    value = 99

    from cmpy.operators import project_elements_up, project_elements_dn
    a = project_elements_up(up_idx, num_dn_states, dn_indices, value)
    for i in a:
        print(i)

    dn_idx = 2
    up_indices = np.array([0,1,2])

    b = project_elements_dn(dn_idx, num_dn_states, up_indices, value)
    for i in b:
        print(i)


if __name__ == "__main__":
    # A=1
    # visualize_states()
    siam()
    # test_project_elements()