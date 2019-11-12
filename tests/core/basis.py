# -*- coding: utf-8 -*-
"""
Created on 11 Nov 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
from numpy.testing import assert_array_equal
from unittest import main, TestCase
from cmpy.core.basis import BasisState, Basis


class TestBasisState(TestCase):

    state1 = BasisState([1, 2], n_sites=3)
    state2 = BasisState([1, 3], n_sites=3)

    def test_array(self):
        assert_array_equal([[0, 0, 1], [0, 1, 0]], self.state1.array())
        assert_array_equal([[0, 0, 1], [0, 1, 1]], self.state2.array())

    def test_n(self):
        self.assertEqual(2, self.state1.n)
        self.assertEqual(3, self.state2.n)

    def test_s(self):
        assert_array_equal([1, 1], self.state1.s)
        assert_array_equal([1, 2], self.state2.s)

    def test_occupation(self):
        assert_array_equal([0, 1, 1], self.state1.occupations)
        assert_array_equal([0, 1, 2], self.state2.occupations)

    def test_interaction(self):
        assert_array_equal([0, 0, 0], self.state1.interactions)
        assert_array_equal([0, 0, 1], self.state2.interactions)

    def test_hopping(self):
        self.assertTrue(self.state1.check_hopping(BasisState([1, 4], 3)))
        self.assertTrue(self.state1.check_hopping(BasisState([2, 2], 3)))
        self.assertTrue(self.state1.check_hopping(BasisState([1, 1], 3)))
        self.assertFalse(self.state1.check_hopping(BasisState([1, 0], 3)))
        self.assertFalse(self.state1.check_hopping(BasisState([2, 1], 3)))

    def test_create(self):
        self.assertEqual(BasisState([3, 2], 3), self.state1.create(1, 0))
        self.assertIsNone(self.state1.create(0, 0))
        self.assertEqual(BasisState([1, 3], 3), self.state1.create(0, 1))
        self.assertIsNone(self.state1.create(1, 1))

    def test_annihilate(self):
        self.assertEqual(BasisState([0, 2], 3), self.state1.annihilate(0, 0))
        self.assertIsNone(self.state1.annihilate(1, 0))
        self.assertEqual(BasisState([1, 0], 3), self.state1.annihilate(1, 1))
        self.assertIsNone(self.state1.annihilate(0, 1))


class TestBasis(TestCase):

    basis = Basis(2, 2)

    def test_basis(self):
        self.assertEqual(self.basis.n, 16)

    def test_get(self):
        states = self.basis.get(1)
        for i in range(4):
            self.assertEqual(states[i].n, 1)

    def test_subbasis(self):
        self.assertIsInstance(self.basis.subbasis(1), Basis)

    def test_index(self):
        s = BasisState([1, 1])
        self.assertIsNotNone(self.basis.index(s))


if __name__ == "__main__":
    main()
