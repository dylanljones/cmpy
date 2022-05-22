# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2022, Dylan Jones

from pytest import mark
from hypothesis import given, strategies as st
import numpy as np
from numpy.testing import assert_array_equal
from cmpy import basis


@given(st.integers(0, int(2**15)))
def test_binstr(num):
    assert basis.binstr(num) == f"{num:b}"
    for width in [0, 5, 10, 15, 20]:
        assert basis.binstr(num, width) == f"{num:0{width}b}"


@given(st.integers(0, int(2**15)))
def test_binarr(num):
    result = np.fromiter(f"{num:b}"[::-1], dtype=np.int64)
    assert_array_equal(basis.binarr(num), result)
    for width in [0, 5, 10, 15, 20]:
        result = np.fromiter(f"{num:0{width}b}"[::-1], dtype=np.int64)
        assert_array_equal(basis.binarr(num, width), result)


@given(st.integers(0, int(2**15)))
def test_binidx(num):
    assert_array_equal(basis.binidx(num), np.where(basis.binarr(num))[0])
    for width in [0, 5, 10, 15, 20]:
        assert_array_equal(
            basis.binidx(num, width), np.where(basis.binarr(num, width))[0]
        )


@given(st.integers(0, int(2**15)), st.integers(0, int(2**15)))
def test_overlap(num1, num2):
    result = np.fromiter(f"{num1 & num2:b}"[::-1], dtype=np.int64)
    assert_array_equal(basis.overlap(num1, num2), result)
    for width in [0, 5, 10, 15, 20]:
        result = np.fromiter(f"{num1 & num2:0{width}b}"[::-1], dtype=np.int64)
        assert_array_equal(basis.overlap(num1, num2, width), result)


@given(st.integers(0, int(2**15)))
def test_occupations(num):
    result = np.fromiter(f"{num:b}"[::-1], dtype=np.int64)
    assert_array_equal(basis.occupations(num), result)
    for width in [0, 5, 10, 15, 20]:
        result = np.fromiter(f"{num:0{width}b}"[::-1], dtype=np.int64)
        assert_array_equal(basis.occupations(num, width), result)


@mark.parametrize(
    "num, pos, result",
    [
        (int("0", 2), 0, int("1", 2)),
        (int("1", 2), 0, None),
        (int("100", 2), 0, int("101", 2)),
        (int("100", 2), 1, int("110", 2)),
        (int("100", 2), 2, None),
        (int("110", 2), 0, int("111", 2)),
        (int("110", 2), 1, None),
        (int("110", 2), 2, None),
    ],
)
def test_create(num, pos, result):
    assert basis.create(num, pos) == result


@mark.parametrize(
    "num, pos, result",
    [
        (int("0", 2), 0, None),
        (int("1", 2), 0, int("0", 2)),
        (int("100", 2), 0, None),
        (int("100", 2), 1, None),
        (int("100", 2), 2, int("000", 2)),
        (int("110", 2), 0, None),
        (int("110", 2), 1, int("100", 2)),
        (int("110", 2), 2, int("010", 2)),
    ],
)
def test_annihilate(num, pos, result):
    assert basis.annihilate(num, pos) == result


@mark.parametrize("n_up", list(range(15)))
@mark.parametrize("n_dn", list(range(15)))
def test_upper_sector(n_up, n_dn):
    num_sites = 15
    # Test spin-up
    res = None if n_up == num_sites else (n_up + 1, n_dn)
    assert basis.upper_sector(n_up, n_dn, basis.UP, num_sites) == res
    # Test spin-down
    res = None if n_dn == num_sites else (n_up, n_dn + 1)
    assert basis.upper_sector(n_up, n_dn, basis.DN, num_sites) == res


@mark.parametrize("n_up", list(range(15)))
@mark.parametrize("n_dn", list(range(15)))
def test_lower_sector(n_up, n_dn):
    # Test spin-up
    res = None if n_up == 0 else (n_up - 1, n_dn)
    assert basis.lower_sector(n_up, n_dn, basis.UP) == res
    # Test spin-down
    res = None if n_dn == 0 else (n_up, n_dn - 1)
    assert basis.lower_sector(n_up, n_dn, basis.DN) == res


@given(st.integers(0, 15), st.booleans())
def test_basis_init(num_sites, init_sectors):
    b = basis.Basis(num_sites, init_sectors)
    assert b.num_spinstates == 2**num_sites
    assert b.size == 2 ** (2 * num_sites)
    if init_sectors:
        for n in range(num_sites):
            assert all(n == f"{x:b}".count("1") for x in b.sectors[n])


@mark.parametrize(
    "num_sites, n, result",
    [
        (2, 0, ["00"]),
        (2, 1, ["01", "10"]),
        (2, 2, ["11"]),
        (2, None, ["00", "01", "10", "11"]),
        (3, 0, ["000"]),
        (3, 1, ["001", "010", "100"]),
        (3, 2, ["011", "101", "110"]),
        (3, 3, ["111"]),
        (3, None, ["000", "001", "010", "011", "100", "101", "110", "111"]),
    ],
)
def test_basis_get_states(num_sites, n, result):
    b = basis.Basis(num_sites)
    states = b.get_states(n)
    assert all(basis.binstr(s, num_sites) == y for s, y in zip(states, result))
