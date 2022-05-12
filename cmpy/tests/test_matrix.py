# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2022, Dylan Jones

import numpy as np
from numpy.testing import assert_allclose
from hypothesis import assume, given, strategies as st
from hypothesis.extra.numpy import arrays
from cmpy import matrix

mats = arrays(
    np.float64,
    st.tuples(
        st.shared(st.integers(2, 10), key="n"),
        st.shared(st.integers(2, 10), key="n"),
    ),
    elements=st.floats(-1e10, +1e10),
)


@given(mats)
def test_reconstruct(arr):
    assume(matrix.is_hermitian(arr))
    vr, xi, vl = matrix.decompose_eig(arr, h=True)
    rec = matrix.reconstruct_eig(vr, xi, vl).astype(np.float64)
    assert_allclose(arr, rec, atol=1e-20)


@given(mats)
def test_reconstruct_qr(arr):
    q, r = matrix.decompose_qr(arr)
    rec = matrix.reconstruct_qr(q, r)
    assert_allclose(arr, rec, atol=1e-20)


@given(mats)
def test_reconstruct_svd(arr):
    u, s, vh = matrix.decompose_svd(arr)
    rec = matrix.reconstruct_svd(u, s, vh)
    assert_allclose(arr, rec, atol=1e-20)
