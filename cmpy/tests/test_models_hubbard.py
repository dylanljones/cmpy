# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2022, Dylan Jones

from numpy.testing import assert_array_equal
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
