# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2020, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import numpy as np
from pytest import mark
from numpy.testing import assert_array_equal
from cmpy import operators


@mark.parametrize("shape", [
    (3, 3),
    (3, 4),
    (4, 3),
])
def test_sparse_init(shape):
    op = operators.SparseOperator(shape)
    assert_array_equal(op.array(), np.zeros(shape))


@mark.parametrize("array", [
    np.random.random((3, 3)),
    np.random.random((3, 4)),
    np.random.random((4, 3)),
])
def test_sparse_set_data(array):
    rows, cols = list(), list()
    data = list()
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] != 0:
                rows.append(i)
                cols.append(j)
                data.append(array[i, j])

    op = operators.SparseOperator(array.shape)
    op.set_data(rows, cols, data)
    assert_array_equal(op.array(), array)


@mark.parametrize("row, col, value", [
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
])
def test_sparse_append_item(row, col, value):
    op = operators.SparseOperator((5, 5))
    op.append(row, col, value)
    arr = op.array()
    assert arr[row, col] == value
    op.append(row, col, value)
    arr = op.array()
    assert arr[row, col] == 2 * value


@mark.parametrize("array", [
    np.random.random((3, 3)),
    np.random.random((3, 4)),
    np.random.random((4, 3)),
])
def test_sparse_append(array):
    op = operators.SparseOperator(array.shape)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] != 0:
                op.append(i, j, array[i, j])
    assert_array_equal(op.array(), array)


@mark.parametrize("array", [
    np.random.random((3, 3)),
    np.random.random((3, 4)),
    np.random.random((4, 3)),
])
def test_sparse_setitem(array):
    op = operators.SparseOperator(array.shape)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] != 0:
                op[i, j] = array[i, j]
    assert_array_equal(op.array(), array)

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] != 0:
                op[i, j] = array[i, j]
    assert_array_equal(op.array(), array)


@mark.parametrize("array", [
    np.random.random((3, 3)),
    np.random.random((3, 4)),
    np.random.random((4, 3)),
])
def test_sparse_getitem(array):
    op = operators.SparseOperator(array.shape)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] != 0:
                op.append(i, j, array[i, j])
    # op = op.array()
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            assert op[i, j] == array[i, j]
