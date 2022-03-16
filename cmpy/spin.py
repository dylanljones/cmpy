# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2022, Dylan Jones

import numpy as np

sigi = np.eye(2)
sigx = np.array([[0, 1], [1, 0]])
sigy = np.array([[0, -1j], [1j, 0]])
sigz = np.array([[1, 0], [0, -1]])
sigp = 0.5 * (sigx + 1j * sigy)
sigm = 0.5 * (sigx - 1j * sigy)

pauli = sigi, sigx, sigy, sigz


def generate_spin_quantum_numbers(s=0.5):
    """Generates the spin quantum numbers `m=(-S, -S+1, ..., +S)` for the spin `S`."""
    return np.arange(-s, s + 0.01)[::-1]


def generate_spin_basis(m_values):
    pairs = np.zeros((len(m_values), len(m_values), 2))
    for i, n in enumerate(m_values):
        for j, m in enumerate(m_values):
            pairs[i, j] = n, m
    return pairs


def apply_sx(n, m, s=0.5):
    r"""Computes the product `<n|S_x|m>`.
    .. math::
        <n|S_x|m> = 1/2 (δ_{n,m+1} + δ_{n+1,m}) \sqrt{S(S+1) - nm}
    """
    n = np.asarray(n)
    m = np.asarray(m)
    delta = np.bitwise_or(n == m + 1, n + 1 == m)
    return 0.5 * delta * np.sqrt(s * (s + 1) - n * m)


def apply_sy(n, m, s=0.5):
    r"""Computes the product `<n|S_y|m>`.
    .. math::
        <n|S_y|m> = -i/2 (δ_{n,m+1} - δ_{n+1,m}) \sqrt{S(S+1) - nm}
    """
    n = np.asarray(n)
    m = np.asarray(m)
    delta1 = np.array(n == m + 1).astype(np.int8)
    delta2 = np.array(n + 1 == m).astype(np.int8)
    return 0.5j * (delta1 - delta2) * np.sqrt(s * (s + 1) - n * m)


def apply_sz(n, m):
    r"""Computes the product `<n|S_z|m>`.
    .. math::
        <n|S_z|m> = δ_{n,m} m
    """
    n = np.asarray(n)
    m = np.asarray(m)
    delta = n == m
    return delta * m


def apply_sp(n, m, s):
    r"""Computes the product `<n|S_+|m>`.
    .. math::
        <n|S_+|m> = δ_{n+1,m} \sqrt{S(S+1) - nm}
    """
    n = np.asarray(n)
    m = np.asarray(m)
    delta = n + 1 == m
    return delta * np.sqrt(s * (s + 1) - n * m)


def apply_sm(n, m, s):
    r"""Computes the product `<n|S_-|m>`.
    .. math::
        <n|S_-|m> = δ_{n,m+1} \sqrt{S(S+1) - nm}
    """
    n = np.asarray(n)
    m = np.asarray(m)
    delta = n == m + 1
    return delta * np.sqrt(s * (s + 1) - n * m)


def construct_sx(s):
    m_values = generate_spin_quantum_numbers(s)
    n, m = np.meshgrid(m_values, m_values)
    return apply_sx(n, m, s)


def construct_sy(s):
    m_values = generate_spin_quantum_numbers(s)
    n, m = np.meshgrid(m_values, m_values)
    return apply_sy(n, m, s)


def construct_sz(s):
    m_values = generate_spin_quantum_numbers(s)
    n, m = np.meshgrid(m_values, m_values)
    return apply_sz(n, m)


def construct_sp(s):
    m_values = generate_spin_quantum_numbers(s)
    n, m = np.meshgrid(m_values, m_values)
    return apply_sp(n, m, s)


def construct_sm(s):
    m_values = generate_spin_quantum_numbers(s)
    n, m = np.meshgrid(m_values, m_values)
    return apply_sm(n, m, s)
