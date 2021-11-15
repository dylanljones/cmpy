# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2021, Dylan Jones

import numpy as np
from scipy import linalg as la
from numpy.lib import scimath
import matplotlib.pyplot as plt
from cmpy import Matrix
from lattpy import Lattice, simple_chain, simple_square
from cmpy.basis import SpinBasis, SpinState, spinstate_label
from cmpy.models import AbstractSpinModel
from cmpy.operators import LinearOperator
from cmpy.models.heisenberg import HeisenbergModel
from cmpy.ed.lanczos import lanczos_coeffs, lanczos_ground_state


def compute_ground_state(ham):
    eigvals, eigvecs = la.eigh(ham, check_finite=False)
    idx = np.argmin(eigvals)
    return eigvals[idx], eigvecs[:, idx]


def ground_state(ham, max_size=30):
    if len(ham) <= max_size:
        return compute_ground_state(ham)
    a_coeffs, b_coeffs = lanczos_coeffs(ham, max_size)
    return lanczos_ground_state(a_coeffs, b_coeffs)


def compare_lanczos(ham, egs):
    egs_l, gs_l = ground_state(ham, max_size=20)
    print("Exact:  ", egs)
    print("Lanczos:", egs_l)
    print("Error:   ", abs(egs - egs_l))


def plot_sz_corr(model, states, gs):
    # compute correlation functions
    deltas = np.arange(0, model.num_sites)
    corr = np.zeros(len(deltas))
    for i, delta in enumerate(deltas):
        corr[i] = sz_correl(states, gs, delta)
    fig, ax = plt.subplots()
    ax.plot(deltas, corr)
    ax.plot(+0.25/deltas, color="k")
    ax.plot(-0.25/deltas, color="k")
    ax.grid()
    plt.show()


def sz_expval(states, gs, pos=0):
    sz = 0.
    for ai, si in zip(gs, states):
        if ai:
            b = (si >> pos) & 1   # Bit at index `pos`
            sign = (-1) ** (b+1)  # Sign +1 if bit is 1, -1 if bit is 0
            sz += sign / 2 * ai * ai
    return sz


def sz_expval2(states, gs, pos=0):
    # Compute |ψ⟩ = S^z|GS⟩
    psi = gs.copy()
    for i, si in enumerate(states):
        b = (si >> pos) & 1   # Bit at index `pos`
        sign = (-1) ** (b+1)  # Sign +1 if bit is 1, -1 if bit is 0
        psi[i] *= sign / 2    # Apply spin of site to state
    # Return scalar product ⟨GS|ψ⟩ = ⟨GS|S^z|GS⟩
    return np.dot(gs, psi)


def sz_expval3(states, gs, pos=0):
    # Construct S^z operator in matrix representation
    size = len(states)
    op = np.zeros((size, size))
    for i, si in enumerate(states):
        b = (si >> pos) & 1   # Bit at index `pos`
        sign = (-1) ** (b+1)  # Sign +1 if bit is 1, -1 if bit is 0
        op[i, i] = sign / 2    # Apply spin of site to state
    # Return product ⟨GS|S^z|GS⟩
    return np.dot(gs, np.dot(op, gs))


class SzOperator(LinearOperator):

    def __init__(self, states, pos=0):
        size = len(states)
        super().__init__((size, size))
        self.states = states
        self.pos = pos

    def _matvec(self, x):
        matvec = np.zeros((self.shape[0], *x.shape[1:]), dtype=x.dtype)
        for i, si in enumerate(self.states):
            b = (si >> self.pos) & 1  # Bit at index `pos`
            sign = (-1) ** (b + 1)    # Sign +1 if bit is 1, -1 if bit is 0
            matvec[i] = sign * x[i] / 2
        return matvec


class SpOperator(LinearOperator):

    def __init__(self, states, pos=0):
        size = len(states)
        super().__init__((size, size))
        self.states = states
        self.pos = pos

    def _matvec(self, x):
        matvec = np.zeros((self.shape[0], *x.shape[1:]), dtype=x.dtype)
        pass
        return matvec


class SmOperator(LinearOperator):

    def __init__(self, states, pos=0):
        size = len(states)
        super().__init__((size, size))
        self.states = states
        self.pos = pos

    def _matvec(self, x):
        matvec = np.zeros((self.shape[0], *x.shape[1:]), dtype=x.dtype)
        pass
        return matvec


def sz_expval4(states, gs, pos=0):
    op = SzOperator(states, pos)
    return np.dot(gs, op @ gs)


def sz_correl(states, gs, delta, j=1):
    res = 0
    for ai, si in zip(gs, states):
        b1 = (si >> 0) & 1                  # Bit at index `0`
        b2 = (si >> delta) & 1              # Bit at index `δ`
        sign = (-1) ** b1 * (-1) ** b2      # Sign +1 if bits are equal, -1 otherwise
        res += ai * ai * sign * j / 4
    return res


def sz_correl2(states, gs, delta, j=1):
    op1 = SzOperator(states, pos=0)
    op2 = SzOperator(states, pos=delta)
    return j * np.dot(gs, op2 @ op1 @ gs)


def main():
    stot = 0
    num_sites = 10
    latt = simple_chain()
    latt.build(num_sites, relative=True, periodic=None)
    model = HeisenbergModel(latt)

    states = model.get_states(stot)
    ham = model.hamiltonian(states=states)
    egs, gs = compute_ground_state(ham)
    # compare_lanczos(ham, egs)

    # compute correlation functions
    deltas = np.arange(0, model.num_sites)
    corr1 = np.zeros(len(deltas))
    corr2 = np.zeros(len(deltas))
    for i, delta in enumerate(deltas):
        corr1[i] = sz_correl(states, gs, delta)
        corr2[i] = sz_correl2(states, gs, delta)

    fig, ax = plt.subplots()
    ax.plot(deltas, corr1)
    ax.plot(deltas, corr2, ls="--")
    ax.plot(+0.25/deltas, color="k")
    ax.plot(-0.25/deltas, color="k")
    ax.grid()
    plt.show()


if __name__ == "__main__":
    main()
