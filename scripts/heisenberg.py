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
from cmpy.basis import SpinBasis


def compute_ground_state(ham):
    eigvals, eigvecs = la.eigh(ham, check_finite=False)
    idx = np.argmin(eigvals)
    return eigvals[idx], eigvecs[:, idx]


def get_chain_neighbors(num_sites, pbc=True):
    if pbc:
        neighbors = np.arange(num_sites) + 1
        neighbors[-1] = 0
    else:
        neighbors = np.arange(num_sites - 1) + 1
    return neighbors


def heisenberg_hamiltonian_data(num_sites, neighbors, states, j=1.):
    for idx1, s1 in enumerate(states):
        for pos1 in range(num_sites):
            try:
                pos2 = neighbors[pos1]
            except IndexError:
                continue
            # Diagonal
            b1 = (s1 >> pos1) & 1           # Bit at index `pos1`
            b2 = (s1 >> pos2) & 1           # Bit at index `pos2`
            sign = (-1) ** b1 * (-1) ** b2  # Sign +1 if bits are equal, -1 otherwise
            yield idx1, idx1, sign * j / 4  # Add diagonal element (sign * 1/4)

            # Off-diagonal
            op = 1 << pos1                  # bit-operator at `pos1`
            occ = s1 & op                   # bit value of bit  at `pos1`
            op2 = 1 << pos2                 # bit-operator at `pos2`
            occ2 = s1 & op2                 # bit value of bit  at `pos2`
            if (occ and not occ2) or (not occ and occ2):
                # Hopping between `pos1` to `pos2` possible
                tmp = s1 ^ op               # Annihilate or create state at `pos1`
                s2 = tmp ^ op2              # create new state with XOR
                idx2 = states.index(s2)     # get index of new state
                yield idx1, idx2, j / 2     # Add off-diagonal element (always 1/2)


class HeisenbergModel:

    def __init__(self, latt, j=1.):
        self.latt = latt
        self.basis = SpinBasis(latt.num_sites)
        self.j = j

    @property
    def num_sites(self):
        return self.basis.num_sites

    def get_states(self, s=None):
        return self.basis.get_states(s)

    def _hamiltonian_data(self, states):
        for idx1, s1 in enumerate(states):
            for pos1 in range(self.num_sites):
                pos2 = self.latt.neighbors(pos1)[0]
                # Diagonal
                b1 = (s1 >> pos1) & 1           # Bit at index `pos1`
                b2 = (s1 >> pos2) & 1           # Bit at index `pos2`
                sign = (-1)**b1 * (-1)**b2      # Sign +1 if bits are equal, -1 otherwise
                yield idx1, idx1, sign * self.j / 4
                # Off-diagonal
                op = 1 << pos1                  # bit-operator at `pos1`
                occ = s1 & op                   # bit value of bit  at `pos1`
                op2 = 1 << pos2                 # bit-operator at `pos2`
                occ2 = s1 & op2                 # bit value of bit  at `pos2`
                if (occ and not occ2) or (not occ and occ2):
                    # Hopping between `pos1` to `pos2` possible
                    tmp = s1 ^ op               # Annihilate or create state at `pos1`
                    s2 = tmp ^ op2              # create new state with XOR
                    idx2 = states.index(s2)     # get index of new state
                    yield idx1, idx2, self.j / 2

    def hamiltonian(self, s=None, states=None):
        if states is None:
            states = self.get_states(s)
        shape = len(states), len(states)
        ham = np.zeros(shape, np.float64)
        for i, j, val in self._hamiltonian_data(states):
            ham[i, j] += val
        return ham


def sz_correl(states, gs, delta, j=1):
    res = 0
    for ai, si in zip(gs, states):
        b1 = (si >> 0) & 1                  # Bit at index `0`
        b2 = (si >> delta) & 1              # Bit at index `δ`
        sign = (-1) ** b1 * (-1) ** b2      # Sign +1 if bits are equal, -1 otherwise
        res += ai * ai * sign * j / 4
    return res


def iter_lanczos_coeffs(ham, size=10):
    # Initial guess of wavefunction
    psi = np.random.uniform(0, 1, size=len(ham))
    # First iteration only contains diagonal coefficient
    a = np.dot(psi, np.dot(ham, psi)) / np.dot(psi, psi)
    yield a, None
    # Compute new wavefunction:
    # |ψ_1⟩ = H |ψ_0⟩ - a_0 |ψ_0⟩
    psi_new = np.dot(ham, psi) - a * psi
    psi_prev = psi
    psi = psi_new
    # Continue iterations
    for n in range(1, size):
        # Compute coefficients a_n, b_n^2
        a = np.dot(psi, np.dot(ham, psi)) / np.dot(psi, psi)
        b2 = np.dot(psi, psi) / np.dot(psi_prev, psi_prev)
        # Compute new wavefunction
        # |ψ_{n+1}⟩ = H |ψ_n⟩ - a_n |ψ_n⟩ - b_n^2 |ψ_{n-1}⟩
        psi_new = np.dot(ham, psi) - a * psi - b2 * psi_prev
        # Save coefficients and update wave functions
        b = scimath.sqrt(b2)
        yield a, b
        psi_prev = psi
        psi = psi_new


def lanczos_coeffs(ham, size=10):
    a_coeffs = list()
    b_coeffs = list()
    for a, b in iter_lanczos_coeffs(ham, size):
        a_coeffs.append(a)
        b_coeffs.append(b)
    # remove None from b_coeffs
    b_coeffs.pop(0)
    return a_coeffs, b_coeffs


def lanczos_matrix(a_coeffs, b_coeffs):
    mat = np.diag(a_coeffs)
    np.fill_diagonal(mat[1:], b_coeffs)
    np.fill_diagonal(mat[:, 1:], b_coeffs)
    return mat


def lanczos_ground_state(a_coeffs, b_coeffs):
    eigvals, eigvecs = la.eigh_tridiagonal(a_coeffs, b_coeffs, select="i", select_range=(0, 2))
    idx = np.argmin(eigvals)
    e_gs = eigvals[idx]
    gs = eigvecs[:, idx]
    return e_gs, gs


def ground_state(ham, max_size=30):
    if len(ham) <= max_size:
        return compute_ground_state(ham)
    a_coeffs, b_coeffs = lanczos_coeffs(ham, max_size)
    return lanczos_ground_state(a_coeffs, b_coeffs)


def main():
    stot = None
    num_sites = 10
    latt = simple_chain()
    latt.build(num_sites, relative=True, periodic=0)
    model = HeisenbergModel(latt)

    print("Building hamiltonian")
    states = model.get_states(stot)
    ham = model.hamiltonian(states=states)

    print("Diagonalizing")
    egs_ref, gs_ref = compute_ground_state(ham)
    print("Diagonalizing2")
    egs, gs = ground_state(ham, max_size=30)
    print("Exact:  ", egs_ref)
    print("Lanczos:", egs)
    print("Error:   ", abs(egs_ref - egs))

    # compute correlation functions
    deltas = np.arange(0, num_sites)
    corr = np.zeros(len(deltas))
    for i, delta in enumerate(deltas):
        corr[i] = sz_correl(states, gs_ref, delta)

    fig, ax = plt.subplots()
    ax.plot(deltas, corr)
    ax.plot(+0.25/deltas, color="k")
    ax.plot(-0.25/deltas, color="k")
    ax.grid()
    plt.show()


if __name__ == "__main__":
    main()
