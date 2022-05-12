# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2022, Dylan Jones

from pytest import mark
from numpy.testing import assert_allclose
import numpy as np
import lattpy as lp
from cmpy.greens import gf0_lehmann
from cmpy.exactdiag import greens_function_lehmann
from cmpy.models import HubbardModel


def tight_binding_hamiltonian(latt, eps=0.0, hop=1.0):
    dmap = latt.dmap()
    data = np.zeros(dmap.size)
    data[dmap.onsite()] = eps
    data[dmap.hopping()] = hop
    return dmap.build_csr(data).toarray()


@mark.parametrize("num_sites", [2, 3, 4, 5])
def test_gf_lehmann_hubbard_non_interacting(num_sites):
    eps = 0.0
    pos = 0
    z = np.linspace(-6, +6, 1001) + 0.05j

    latt = lp.finite_hypercubic(num_sites)
    neighbors, _ = latt.neighbor_pairs(unique=True)
    model = HubbardModel(num_sites, neighbors, eps=eps, hop=1.0)

    ham0 = tight_binding_hamiltonian(latt, eps=model.eps, hop=model.hop)
    gf0_z = gf0_lehmann(ham0, z=z)[:, pos]

    gf_meas = greens_function_lehmann(model, z, beta=10, pos=pos, occ=False)
    gf_z = gf_meas.gf

    assert_allclose(gf_z, gf0_z, rtol=1e-4, atol=1e-3)


@mark.parametrize("num_sites", [2, 3, 4, 5])
def test_gf_lehmann_hubbard_atomic_limit(num_sites):
    u = 2.0
    pos = 0
    z = np.linspace(-4, +4, 1001) + 0.05j

    latt = lp.finite_hypercubic(num_sites)
    neighbors, _ = latt.neighbor_pairs(unique=True)
    model = HubbardModel(num_sites, neighbors, inter=u, mu=u / 2, hop=0.0)

    gf_meas = greens_function_lehmann(model, z, beta=10, pos=pos, occ=False)
    gf_z = gf_meas.gf

    c = len(z) // 2
    gf_neg = -gf_z.imag[:c]
    ww_neg = z.real[:c]
    gf_pos = -gf_z.imag[c:]
    ww_pos = z.real[c:]
    # z < 0
    peak0 = np.argmax(gf_neg)
    energy0 = ww_neg[peak0]
    # z > 0
    peak1 = np.argmax(gf_pos)
    energy1 = ww_pos[peak1]

    assert abs(energy0 - (-u / 2)) < 0.1
    assert abs(energy1 - (+u / 2)) < 0.1
