# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2022, Dylan Jones

import numpy as np
from hypothesis import given, strategies as st
from numpy.testing import assert_array_equal
from cmpy.basis import Basis
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
