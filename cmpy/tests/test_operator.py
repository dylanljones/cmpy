# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2022, Dylan Jones

import numpy as np
from pytest import mark
from hypothesis import given, strategies as st
from numpy.testing import assert_array_equal
from cmpy.basis import Basis, UP, DN
from cmpy import operators


@given(st.integers(0, 5))
def test_project_up(up_idx):
    sec = Basis(5).get_sector()
    indices = [i for i, state in enumerate(sec.states) if state.up == up_idx]
    result = operators.project_up(up_idx, sec.num_dn, np.arange(sec.num_dn))
    assert_array_equal(indices, result)


@given(st.integers(0, 5))
def test_project_dn(dn_idx):
    sec = Basis(5).get_sector()
    indices = [i for i, state in enumerate(sec.states) if state.dn == dn_idx]
    result = operators.project_dn(dn_idx, sec.num_dn, np.arange(sec.num_up))
    assert_array_equal(indices, result)


@mark.parametrize("num_sites", [2, 3, 4, 5])
@mark.parametrize("sigma", [UP, DN])
def test_creation_annihilation_adjoint(num_sites, sigma):
    basis = Basis(num_sites)
    for n_up, n_dn in basis.iter_fillings():
        sector = basis.get_sector(n_up, n_dn)
        sector_p1 = basis.upper_sector(n_up, n_dn, sigma)
        if sector_p1 is not None:
            for pos in range(num_sites):
                cop_dag = operators.CreationOperator(sector, sector_p1, pos, sigma)
                cop = operators.AnnihilationOperator(sector_p1, sector, pos, sigma)
                assert_array_equal(cop.toarray(), cop_dag.toarray().T.conj())
