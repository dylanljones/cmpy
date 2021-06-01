#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from lattpy import Lattice
from cmpy.models import SingleImpurityAndersonModel
import gftool as gt
import gftool.siam

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


def siam():
    import matplotlib.pyplot as plt

    U =0.0
    eps = np.array([1.0, 1.0, 1.0])-0.5
    V = np.array([1.0,1.0])
    siam = SingleImpurityAndersonModel(u=U, eps_imp=eps[0], eps_bath=eps[1:], v=V)
    # zz = np.linspace(-2,2,50) + 0.001j
    # gf0_loc_z = partial(gt.siam.gf0_loc_z, e_onsite=eps[0], e_bath=eps[1:], hopping_sqr=abs(V) ** 2)


    # siam.impurity_gf(zz, 10)
    gs = siam.get_total_gs()

    tt, overlaps_gr = siam.t_evo_gr(gs, 0, 100, 1000)
    tt, overlaps_ls = siam.t_evo_ls(gs, 0, 100, 1000)
    delta = 1e-4
    eta = -np.log(delta) / tt[-1]
    ww = np.linspace(-4, 4, num=5001) + 1j * eta

    print(f"particle sector: {gs.nup}, {gs.ndn}")
    g_ret = overlaps_gr + overlaps_ls
    # gf0_ret_t = siam_weh.gf0_loc_ret_t(tt, eps[0], eps[1:], V)
    # gf_0up_ww = gt.fourier.tt2z(tt, gf_0_ret_t, ww)

    gf0_ww = gt.siam.gf0_loc_z(ww, eps[0], e_bath=eps[1:], hopping_sqr=abs(V) ** 2)
    # plt.plot(tt, overlaps_gr)
    # plt.plot(tt, overlaps_ls)
    # plt.plot(tt, g_ret)


    g_ret_ww = gt.fourier.tt2z(tt, g_ret, ww)
    plt.plot(ww, -g_ret_ww.imag, label=f"tevo, U={U}")
    plt.plot(ww, -gf0_ww.imag, label="non-int Weh")
    plt.legend()
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