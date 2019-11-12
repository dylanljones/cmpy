# -*- coding: utf-8 -*-
"""
Created on 11 Nov 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from unittest import main, TestCase
from cmpy.core import math


class TestMath(TestCase):

    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.array([[1, 0], [0, -1]])

    def test_check_hermitian(self):
        self.assertTrue(math.check_hermitian(np.array([[1, 1j], [-1j, 1]])))
        self.assertFalse(math.check_hermitian(np.array([[1, 1], [-1, 1]])))

    def test_decompose(self):
        v_inv, x, v = math.decompose(self.sx)
        eigvecs = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
        assert_array_almost_equal([1, -1], x)
        assert_array_almost_equal(eigvecs, v)
        assert_array_almost_equal(np.linalg.inv(eigvecs), v_inv)

        v_inv, x, v = math.decompose(self.sy)
        eigvals = np.array([1, -1], dtype="complex")
        eigvecs = np.array([[-1j, 1], [1, -1j]]) / np.sqrt(2)
        assert_array_almost_equal(eigvals, x)
        assert_array_almost_equal(eigvecs, v)
        assert_array_almost_equal(eigvecs.conj().T, v_inv)

        v_inv, x, v = math.decompose(self.sz)
        eigvecs = np.array([[1, 0], [0, 1]])
        assert_array_almost_equal([1, -1], x)
        assert_array_almost_equal(eigvecs, v)
        assert_array_almost_equal(np.linalg.inv(eigvecs), v_inv)

    def test_reconstruct(self):
        v_inv, x, v = math.decompose(self.sx)
        assert_array_equal(self.sx, math.reconstruct(v_inv, x, v))
        v_inv, x, v = math.decompose(self.sy)
        assert_array_equal(self.sy, math.reconstruct(v_inv, x, v))
        v_inv, x, v = math.decompose(self.sz)
        assert_array_equal(self.sz, math.reconstruct(v_inv, x, v))

    def test_partition(self):
        e = np.arange(3)
        self.assertEqual(1.5032147244080551, math.partition(e, 1.))


if __name__ == "__main__":
    main()
