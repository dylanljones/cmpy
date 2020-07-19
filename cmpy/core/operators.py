# coding: utf-8
"""
Created on 07 Jul 2020
Author: Dylan Jones
"""
import abc
import numpy as np
import scipy.linalg as la
from typing import Callable, Iterable, List
from scipy.sparse import linalg as sla
from bisect import bisect_left
from .matrix import Matrix, is_hermitian
from .basis import UP, SPIN_CHARS, binstr, occupations, overlap


# =========================================================================
# Linear Operator
# =========================================================================


def sum_weighted(weights, arr):
    return np.sum(arr[:weights.size] * weights)


def project_up(up_idx, num_dn_states, dn_indices):
    return up_idx * num_dn_states + dn_indices


def project_dn(dn_idx, num_up_states, up_indices):
    return up_indices * num_up_states + dn_idx


def apply_projected_up(matvec, x, num_dn_states, up_idx, dn_indices, value, target=None):
    if value:
        origins = project_up(up_idx, num_dn_states, dn_indices)
        targets = origins if target is None else project_up(target, num_dn_states, dn_indices)
        matvec[targets] += value * x[origins]


def apply_projected_dn(matvec, x, num_dn_states, dn_idx, up_indices, value, target=None):
    if value:
        origins = project_dn(dn_idx, num_dn_states, up_indices)
        targets = origins if target is None else project_dn(target, num_dn_states, up_indices)
        matvec[targets] += value * x[origins]


class LinearOperator(abc.ABC, sla.LinearOperator):

    def __init__(self, shape, dtype=None):
        super().__init__(shape=shape, dtype=dtype)

    def __repr__(self):
        return f"{self.__class__.__name__}(shape: {self.shape}, dtype: {self.dtype})"

    @abc.abstractmethod
    def _matvec(self, x):
        pass

    def matmat(self, x=None):
        """ Matrix-matrix multiplication.

        Performs the operation y=A*X where A is an MxN linear operator and X dense N*K matrix or ndarray.

        Parameters
        ----------
        x: array_like, optional
            The other matrix of the matrix-matrix product. The default is the identity matrix.
        """
        return super().matmat(np.eye(*self.shape) if x is None else x)

    def matrix(self, x=None):
        """ Matrix: Returns a `Matrix`-instance of a matrix-matrix multiplication """
        return Matrix(self.matmat(x))


# =========================================================================
# Hamiltonian
# =========================================================================


def _hopping_candidates(num_sites, state, pos):
    results = []
    op = 1 << pos
    occ = state & op

    tmp = state ^ op  # Annihilate or create electron at `pos`
    for pos2 in range(num_sites):
        if pos >= pos2:
            continue
        op2 = (1 << pos2)
        occ2 = state & op2
        # Hopping from `pos` to `pos2` possible
        if occ and not occ2:
            new = tmp ^ op2
            results.append((pos2, new))
        # Hopping from `pos2` to `pos` possible
        elif not occ and occ2:
            new = tmp ^ op2
            results.append((pos2, new))

    return results


def _ordering_phase(state, pos1, pos2=0):
    if pos1 == pos2:
        return 0
    i0, i1 = sorted([pos1, pos2])
    particles = binstr(state)[i0 + 1:i1].count("1")
    return +1 if particles % 2 == 0 else -1


def _calc_hopping(num_sites, states, pos, hopping):
    for i, state in enumerate(states):
        for pos2, new in _hopping_candidates(num_sites, state, pos):
            try:
                t = hopping(pos, pos2)
            except TypeError:
                t = hopping
            if t:
                j = bisect_left(states, new)
                sign = _ordering_phase(state, pos, pos2)
                value = sign * t
                yield i, j, value


def apply_onsite_energy(matvec, x, up_states, dn_states, eps):
    num_dn = len(dn_states)
    all_up, all_dn = np.arange(len(up_states)), np.arange(num_dn)

    for up_idx, up in enumerate(up_states):
        weights = occupations(up)
        energy = np.sum(eps[:weights.size] * weights)
        apply_projected_up(matvec, x, num_dn, up_idx, all_dn, energy)

    for dn_idx, dn in enumerate(dn_states):
        weights = occupations(dn)
        energy = np.sum(eps[:weights.size] * weights)
        apply_projected_dn(matvec, x, num_dn, dn_idx, all_up, energy)


def apply_interaction(matvec, x, up_states, dn_states, u):
    num_dn = len(dn_states)
    for up_idx, up in enumerate(up_states):
        for dn_idx, dn in enumerate(dn_states):
            weights = overlap(up, dn)
            interaction = np.sum(u[:weights.size] * weights)
            apply_projected_up(matvec, x, num_dn, up_idx, dn_idx, interaction)


def apply_site_hopping(matvec, x, up_states, dn_states, num_sites, hopping: (Callable, Iterable, float), pos):
    num_dn = len(dn_states)
    all_up, all_dn = np.arange(len(up_states)), np.arange(num_dn)

    for up_idx, target, amp in _calc_hopping(num_sites, up_states, pos, hopping):
        apply_projected_up(matvec, x, num_dn, up_idx, all_dn, amp, target=target)

    for dn_idx, target, amp in _calc_hopping(num_sites, dn_states, pos, hopping):
        apply_projected_dn(matvec, x, num_dn, dn_idx, all_up, amp, target=target)


def apply_hopping(matvec, x, up_states, dn_states, num_sites, hopping=lambda i, j: int(abs(i - j) == 1)):
    for pos in range(num_sites):
        apply_site_hopping(matvec, x, up_states, dn_states, num_sites, hopping, pos)


class HamiltonOperator(LinearOperator):

    def __init__(self, model, sector):
        self.sector = sector
        self.model = model
        shape = (sector.size, sector.size)
        super().__init__(shape, dtype=np.complex)

    @property
    def up_states(self):
        return self.sector.up_states

    @property
    def dn_states(self):
        return self.sector.dn_states

    @property
    def labels(self):
        return self.sector.labels

    def apply_onsite_energy(self, matvec, x, eps):
        apply_onsite_energy(matvec, x, self.up_states, self.dn_states, eps)

    def apply_interaction(self, matvec, x, u):
        apply_interaction(matvec, x, self.up_states, self.dn_states, u)

    def apply_hopping(self, matvec, x, hopping):
        apply_hopping(matvec, x, self.up_states, self.dn_states, self.sector.num_sites, hopping)

    def apply_site_hopping(self, matvec, x, hopping, pos):
        apply_site_hopping(matvec, x, self.up_states, self.dn_states, self.sector.num_sites, hopping, pos)

    def _matvec(self, x):
        matvec = np.zeros_like(x)
        self.model.build_matvec(matvec, x.copy(), self)
        return matvec

    def eig(self, check_hermitian=True):
        """ Calculate eigenvalues and -vectors of the Hamiltonian.

        Parameters
        ----------
        check_hermitian: bool, optional
            If True and the instance of the the mnatrix is hermitian, np.eigh is used as eigensolver.

        Returns
        -------
        eigenvalues: np.ndarray
            eigenvalues of the matrix
        eigenvectors: np.ndarray
            eigenvectors of the matrix
        """
        mat = self.matmat()
        if check_hermitian and is_hermitian(mat):
            return la.eigh(mat)
        else:
            return la.eig(mat)

    def eigh(self):
        """ Calculate eigenvalues and -vectors of the hermitian matrix.

        Returns
        -------
        eigenvalues: np.ndarray
            eigenvalues of the matrix
        eigenvectors: np.ndarray
            eigenvectors of the matrix
        """
        mat = self.matmat()
        assert is_hermitian(mat)
        return la.eigh(mat)

    def show(self, show=True, x=None, **kwargs):
        matrix = self.matrix(x)
        return matrix.show(show=show, ticklabels=self.labels, **kwargs)


# =========================================================================
# Creation- and Annihilation-Operators
# =========================================================================


class CreationOperator(LinearOperator):

    def __init__(self, sector, sector_p1, pos=0, sigma=UP):
        self.pos = pos
        self.sigma = sigma
        self.sector = sector
        self.sector_p1 = sector_p1
        dim_origin = self.sector.size
        if sigma == UP:
            dim_target = sector_p1.num_up_states * sector.num_dn_states
        else:
            dim_target = sector_p1.num_dn_states * sector.num_up_states
        super().__init__(shape=(dim_target, dim_origin), dtype=np.complex)

    def __repr__(self):
        name = f"{self.__class__.__name__}_{self.pos}{SPIN_CHARS[self.sigma]}"
        return f"{name}(shape: {self.shape}, dtype: {self.dtype})"

    def _build_up(self, x):
        op = 1 << self.pos
        num_dn = len(self.sector.dn_states)
        all_dn = np.arange(num_dn)
        size = len(self.sector_p1.up_states) * num_dn
        operator = np.zeros((size, *x.shape[1:]), dtype=x.dtype)
        for up_idx, up in enumerate(self.sector.up_states):
            if not (up & op):
                new = up ^ op
                idx_new = bisect_left(self.sector_p1.up_states, new)
                apply_projected_up(operator, x, num_dn, up_idx, all_dn, value=1, target=idx_new)
        return operator

    def _build_dn(self, x):
        op = 1 << self.pos
        num_up = self.sector.num_up_states
        num_dn = self.sector.num_dn_states
        all_up = np.arange(num_up)
        size = len(self.sector_p1.dn_states) * num_up
        operator = np.zeros((size, *x.shape[1:]), dtype=x.dtype)
        for dn_idx, dn in enumerate(self.sector.dn_states):
            if not (dn & op):
                new = dn ^ op
                idx_new = bisect_left(self.sector_p1.fn_states, new)
                apply_projected_dn(operator, x, num_dn, dn_idx, all_up, value=1, target=idx_new)
        return operator

    def _matvec(self, x):
        if self.sigma == UP:
            return self._build_up(x)
        else:
            return self._build_dn(x)
