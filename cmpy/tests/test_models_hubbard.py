# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2022, Dylan Jones

from pytest import mark
from numpy.testing import assert_array_equal
import lattpy as lp
from cmpy.matrix import is_hermitian
from cmpy.models import HubbardModel


def test_hubbard_sector_1_1():
    neighbors = [[0, 1]]
    model = HubbardModel(2, neighbors, inter=2.0, eps=1.0, hop=1.0)

    expected = [
        [4.0, 1.0, 1.0, 0.0],
        [1.0, 2.0, 0.0, 1.0],
        [1.0, 0.0, 2.0, 1.0],
        [0.0, 1.0, 1.0, 4.0],
    ]
    ham = model.hamiltonian(1, 1)
    assert_array_equal(ham, expected)


@mark.parametrize("num_sites", [1, 2, 3, 4, 5, 6])
@mark.parametrize("u", [0.0, 1.0, 2.0])
def test_hamiltonian_hermitian_1d(num_sites, u):
    latt = lp.finite_hypercubic(num_sites)
    neighbors, _ = latt.neighbor_pairs(unique=True)
    model = HubbardModel(num_sites, neighbors, inter=u, mu=u / 2, hop=1.0)
    for n_up, n_dn in model.basis.iter_fillings():
        ham = model.hamiltonian(n_up, n_dn)
        assert is_hermitian(ham)


@mark.parametrize("num_sites", [1, 2, 3, 4, 5, 6])
@mark.parametrize("u", [0.0, 1.0, 2.0])
def test_hamiltonian_hermitian_1d_periodic(num_sites, u):
    latt = lp.finite_hypercubic(num_sites, periodic=True)
    neighbors, _ = latt.neighbor_pairs(unique=True)
    model = HubbardModel(num_sites, neighbors, inter=u, mu=u / 2, hop=1.0)
    for n_up, n_dn in model.basis.iter_fillings():
        ham = model.hamiltonian(n_up, n_dn)
        assert is_hermitian(ham)


@mark.parametrize("u", [0.0, 1.0, 2.0])
def test_hamiltonian_hermitian_2d(u):
    size = 2
    latt = lp.finite_hypercubic((size, size))
    neighbors, _ = latt.neighbor_pairs(unique=True)
    model = HubbardModel(latt.num_sites, neighbors, inter=u, mu=u / 2, hop=1.0)
    for n_up, n_dn in model.basis.iter_fillings():
        ham = model.hamiltonian(n_up, n_dn)
        assert is_hermitian(ham)
