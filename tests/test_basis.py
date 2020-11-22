# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2020, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

from pytest import mark
from numpy.testing import assert_array_equal
from cmpy import basis


@mark.parametrize("num, width, result", [
    (0, 0, "0"),
    (1, 0, "1"),
    (2, 0, "10"),
    (5, 0, "101"),
    (1, 3, "001"),
    (2, 3, "010"),
    (5, 3, "101"),
    (5, 4, "0101"),
    (5, None, "101")
])
def test_binstr(num, width, result):
    assert basis.binstr(num, width) == result


@mark.parametrize("num, width, result", [
    (0, 0, [0]),
    (1, 0, [1]),
    (2, 0, [0, 1]),
    (5, 0, [1, 0, 1]),
    (1, 3, [1, 0, 0]),
    (2, 3, [0, 1, 0]),
    (5, 3, [1, 0, 1]),
    (5, 4, [1, 0, 1, 0]),
    (5, None, [1, 0, 1])
])
def test_binarr(num, width, result):
    assert_array_equal(basis.binarr(num, width), result)


@mark.parametrize("num, width, result", [
    (0, 0, []),
    (1, 0, [0]),
    (2, 0, [1]),
    (5, 0, [0, 2]),
    (1, 3, [0]),
    (2, 3, [1]),
    (5, 3, [0, 2]),
    (5, 4, [0, 2]),
    (5, None, [0, 2])
])
def test_binidx(num, width, result):
    assert_array_equal(basis.binidx(num, width), result)


@mark.parametrize("num1, num2, width, result", [
    (int("0", 2), int("0", 2), 0, [0]),
    (int("1", 2), int("0", 2), 0, [0]),
    (int("0", 2), int("1", 2), 0, [0]),
    (int("1", 2), int("1", 2), 0, [1]),
    (int("11", 2), int("10", 2), 0, [0, 1]),
    (int("01", 2), int("10", 2), 0, [0]),
    (int("01", 2), int("10", 2), 2, [0, 0]),
    (int("11", 2), int("11", 2), 0, [1, 1]),
])
def test_overlap(num1, num2, width, result):
    assert_array_equal(basis.overlap(num1, num2, width), result)


@mark.parametrize("num, width, result", [
    (0, 0, [0]),
    (1, 0, [1]),
    (2, 0, [0, 1]),
    (5, 0, [1, 0, 1]),
    (1, 3, [1, 0, 0]),
    (2, 3, [0, 1, 0]),
    (5, 3, [1, 0, 1]),
    (5, 4, [1, 0, 1, 0]),
    (5, None, [1, 0, 1])
])
def test_occupations(num, width, result):
    assert_array_equal(basis.binarr(num, width), result)


@mark.parametrize("num, pos, result", [
    (int("0", 2), 0, int("1", 2)),
    (int("1", 2), 0, None),
    (int("100", 2), 0, int("101", 2)),
    (int("100", 2), 1, int("110", 2)),
    (int("100", 2), 2, None),
    (int("110", 2), 0, int("111", 2)),
    (int("110", 2), 1, None),
    (int("110", 2), 2, None)
])
def test_create(num, pos, result):
    assert basis.create(num, pos) == result


@mark.parametrize("num, pos, result", [
    (int("0", 2), 0, None),
    (int("1", 2), 0, int("0", 2)),
    (int("100", 2), 0, None),
    (int("100", 2), 1, None),
    (int("100", 2), 2, int("000", 2)),
    (int("110", 2), 0, None),
    (int("110", 2), 1, int("100", 2)),
    (int("110", 2), 2, int("010", 2))
])
def test_annihilate(num, pos, result):
    assert basis.annihilate(num, pos) == result


@mark.parametrize("num_sites, init_sectors", [
    (2, False),
    (2, True),
    (3, False),
    (3, True),
    (20, False),
    (20, True),
])
def test_basis_init(num_sites, init_sectors):
    b = basis.Basis(num_sites, init_sectors)
    assert b.num_spinstates == 2 ** num_sites
    assert b.size == 2 ** (2 * num_sites)
    if init_sectors:
        for n in range(num_sites):
            assert all(n == f"{x:b}".count("1") for x in b.sectors[n])


@mark.parametrize("num_sites, n, result", [
    (2, 0, ["00"]),
    (2, 1, ["01", "10"]),
    (2, 2, ["11"]),
    (2, None, ["00", "01", "10", "11"]),
    (3, 0, ["000"]),
    (3, 1, ["001", "010", "100"]),
    (3, 2, ["011", "101", "110"]),
    (3, 3, ["111"]),
    (3, None, ["000", "001", "010", "011", "100", "101", "110", "111"]),
])
def test_basis_get_states(num_sites, n, result):
    b = basis.Basis(num_sites)
    states = b.get_states(n)
    assert all(basis.binstr(s, num_sites) == y for s, y in zip(states, result))
