# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2021, Dylan Jones

from pytest import mark
import math
import numpy as np
from numpy.testing import assert_array_almost_equal
from cmpy import spin

# S=0.5
sx_05 = 0.5 * np.array([[0, 1], [1, 0]])
sy_05 = -0.5j * np.array([[0, 1], [-1, 0]])
sz_05 = 0.5 * np.array([[1, 0], [0, -1]])
sp_05 = np.array([[0, 1], [0, 0]])
sm_05 = np.array([[0, 0], [1, 0]])

# S=1.0
sqrt2 = math.sqrt(2)
sx_10 = 1/sqrt2 * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
sy_10 = -1j/sqrt2 * np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]])
sz_10 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
sp_10 = sqrt2 * np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
sm_10 = sqrt2 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])

# S=1.5
s3 = math.sqrt(3)
sx_15 = 0.5*np.array([[0, s3, 0, 0], [s3, 0, 2, 0], [0, 2, 0, s3], [0, 0, s3, 0]])
sy_15 = -0.5j*np.array([[0, s3, 0, 0], [-s3, 0, 2, 0], [0, -2, 0, s3], [0, 0, -s3, 0]])
sz_15 = np.array([[1.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, -0.5, 0], [0, 0, 0, -1.5]])
sp_15 = np.array([[0, s3, 0, 0], [0, 0, 2, 0], [0, 0, 0, s3], [0, 0, 0, 0]])
sm_15 = np.array([[0, 0, 0, 0], [s3, 0, 0, 0], [0, 2, 0, 0], [0, 0, s3, 0]])


@mark.parametrize("s, expected", [(0.5, sx_05), (1.0, sx_10), (1.5, sx_15)])
def test_construct_sx(s, expected):
    result = spin.construct_sx(s)
    assert_array_almost_equal(expected, result)


@mark.parametrize("s, expected", [(0.5, sy_05), (1.0, sy_10), (1.5, sy_15)])
def test_construct_sy(s, expected):
    result = spin.construct_sy(s)
    assert_array_almost_equal(expected, result)


@mark.parametrize("s, expected", [(0.5, sz_05), (1.0, sz_10), (1.5, sz_15)])
def test_construct_sz(s, expected):
    result = spin.construct_sz(s)
    assert_array_almost_equal(expected, result)


@mark.parametrize("s, expected", [(0.5, sp_05), (1.0, sp_10), (1.5, sp_15)])
def test_construct_sp(s, expected):
    result = spin.construct_sp(s)
    assert_array_almost_equal(expected, result)


@mark.parametrize("s, expected", [(0.5, sm_05), (1.0, sm_10), (1.5, sm_15)])
def test_construct_sm(s, expected):
    result = spin.construct_sm(s)
    assert_array_almost_equal(expected, result)
