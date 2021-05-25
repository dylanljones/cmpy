# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""This module contains tools for working with linear operators in sparse format."""

import abc
import numpy as np
import scipy.linalg as la
from bisect import bisect_left
from scipy.sparse import linalg as sla
from scipy.sparse import csr_matrix
from typing import Callable, Iterable
from .basis import binstr, occupations, overlap, UP, SPIN_CHARS
from .matrix import Matrix, is_hermitian, Decomposition


__all__ = ["LinearOperator", "SparseOperator", "TimeEvolutionOperator", "CreationOperator",
           "project_up", "project_dn", "project_elements_up", "project_elements_dn",
           "project_interaction", "project_onsite_energy",
           "project_site_hopping", "project_hopping"]


def project_up(up_idx: (int, np.ndarray),
               num_dn_states: int,
               dn_indices: (int, np.ndarray)) -> np.ndarray:
    """Projects spin-up states onto the full basis(-sector).

    Parameters
    ----------
    up_idx: int or np.ndarray
        The index/indices for the projection.
    num_dn_states: int
        The total number of spin-down states of the basis(-sector).
    dn_indices: ndarray
        An array of the indices of all spin-down states in the basis(-sector).
    """
    return up_idx * num_dn_states + dn_indices


def project_dn(dn_idx: (int, np.ndarray),
               num_dn_states: int,
               up_indices: (int, np.ndarray)) -> np.ndarray:
    """Projects spin-down states onto the full basis(-sector).

    Parameters
    ----------
    dn_idx: int or np.ndarray
        The index/indices for the projection.
    num_dn_states: int
        The total number of spin-down states of the basis(-sector).
    up_indices: ndarray
        An array of the indices of all spin-up states in the basis(-sector).
    """
    return up_indices * num_dn_states + dn_idx


def project_elements_up(up_idx: (int, np.ndarray),
                        num_dn_states: int,
                        dn_indices: (int, np.ndarray),
                        value: (complex, float),
                        target: (int, np.ndarray) = None):
    """Projects a value for spin-up states onto the elements of the full basis(-sector).

    Parameters
    ----------
    up_idx: int or np.ndarray
        The index/indices for the projection.
    num_dn_states: int
        The total number of spin-down states of the basis(-sector).
    dn_indices: int or np.ndarray
        An array of the indices of all spin-down states in the basis(-sector).
    value: float or complex
        The value to project.
    target: int or np.ndarray, optional
        The target index/indices for the projection. This is only needed
        for non-diagonal elements.

    Yields
    -------
    row: int
        The row-index of the element.
    col: int
        The column-index of the element.
    value: float or complex
        The value of the matrix-element.
    """
    if not value:
        return

    origins = project_up(up_idx, num_dn_states, dn_indices)
    if target is None:
        targets = origins
    else:
        targets = project_up(target, num_dn_states, dn_indices)

    if isinstance(origins, int):
        yield origins, targets, value
    else:
        for row, col in zip(origins, targets):
            yield row, col, value


def project_elements_dn(dn_idx: (int, np.ndarray),
                        num_dn_states: int,
                        up_indices: (int, np.ndarray),
                        value: (complex, float),
                        target: (int, np.ndarray) = None):
    """Projects a value for spin-down states onto the elements of the full basis(-sector).

    Parameters
    ----------
    dn_idx: int or ndarray
        The index/indices for the projection.
    num_dn_states: int
        The total number of spin-down states of the basis(-sector).
    up_indices: int or np.ndarray
        An array of the indices of all spin-up states in the basis(-sector).
    value: float or complex
        The value to project.
    target: int or np.ndarray, optional
        The target index/indices for the projection. This is only needed
        for non-diagonal elements.

    Yields
    -------
    row: int
        The row-index of the element.
    col: int
        The column-index of the element.
    value: float or complex
        The value of the matrix-element.
    """
    if not value:
        return

    origins = project_dn(dn_idx, num_dn_states, up_indices)
    if target is None:
        targets = origins
    else:
        targets = project_dn(target, num_dn_states, up_indices)

    if isinstance(origins, int):
        yield origins, targets, value
    else:
        for row, col in zip(origins, targets):
            yield row, col, value


# =========================================================================
# Interacting Hamiltonian projectors
# =========================================================================


def project_onsite_energy(up_states, dn_states, eps):
    num_dn = len(dn_states)
    all_up, all_dn = np.arange(len(up_states)), np.arange(num_dn)

    for up_idx, up in enumerate(up_states):
        weights = occupations(up)
        energy = np.sum(eps[:weights.size] * weights)
        yield from project_elements_up(up_idx, num_dn, all_dn, energy)

    for dn_idx, dn in enumerate(dn_states):
        weights = occupations(dn)
        energy = np.sum(eps[:weights.size] * weights)
        yield from project_elements_dn(dn_idx, num_dn, all_up, energy)


def project_interaction(up_states, dn_states, u):
    num_dn = len(dn_states)
    for up_idx, up in enumerate(up_states):
        for dn_idx, dn in enumerate(dn_states):
            weights = overlap(up, dn)
            interaction = np.sum(u[:weights.size] * weights)
            yield from project_elements_up(up_idx, num_dn, dn_idx, interaction)


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


def _compute_hopping(num_sites, states, pos, hopping):
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


def project_site_hopping(up_states, dn_states, num_sites: int,
                         hopping: (Callable, Iterable, float), pos: int):
    num_dn = len(dn_states)
    all_up, all_dn = np.arange(len(up_states)), np.arange(num_dn)

    for up_idx, target, amp in _compute_hopping(num_sites, up_states, pos, hopping):
        yield from project_elements_up(up_idx, num_dn, all_dn, amp, target=target)

    for dn_idx, target, amp in _compute_hopping(num_sites, dn_states, pos, hopping):
        yield from project_elements_dn(dn_idx, num_dn, all_up, amp, target=target)


def project_hopping(up_states, dn_states, num_sites, hopping: (Callable, Iterable, float)):
    for pos in range(num_sites):
        yield from project_site_hopping(up_states, dn_states, num_sites, hopping, pos)


# =========================================================================
# Linear Operators
# =========================================================================


class LinearOperator(sla.LinearOperator, abc.ABC):
    """Abstract base class for linear operators.

    Turns any class that imlements the `_matvec`- or `_matmat`-method and
    turns it into an object that behaves like a linear operator.

    Abstract Methods
    ----------------
    _matvec(v): Matrix-vector multiplication.
        Performs the operation y=A*v where A is an MxN linear operator and
        v is a column vector or 1-d array.
        Implementing _matvec automatically implements _matmat (using a naive algorithm).

    _matmat(X): Matrix-matrix multiplication.
        Performs the operation Y=A*X where A is an MxN linear operator and
        X is a NxM matrix.
        Implementing _matmat automatically implements _matvec (using a naive algorithm).

    _adjoint(): Hermitian adjoint.
        Returns the Hermitian adjoint of self, aka the Hermitian conjugate or Hermitian transpose.
        For a complex matrix, the Hermitian adjoint is equal to the conjugate transpose.
        Can be abbreviated self.H instead of self.adjoint().
        As with _matvec and _matmat, implementing either _rmatvec or _adjoint implements the
        other automatically. Implementing _adjoint is preferable
    """

    def __init__(self, shape, dtype=None):
        sla.LinearOperator.__init__(self, shape=shape, dtype=dtype)
        abc.ABC.__init__(self)

    def __repr__(self):
        return f"{self.__class__.__name__}(shape: {self.shape}, dtype: {self.dtype})"

    def array(self) -> np.ndarray:
        """Returns the `LinearOperator` in form of a dense array."""
        x = np.eye(self.shape[1], dtype=self.dtype)
        return self.matmat(x)

    def matrix(self) -> Matrix:
        """Returns the `LinearOperator` in form of a dense `Matrix`-object."""
        return Matrix(self.array())

    def eig(self, check_hermitian=True):
        """ Calculate eigenvalues and -vectors of the Hamiltonian.

        Parameters
        ----------
        check_hermitian: bool, optional
            If `True` and the instance of the the mnatrix is hermitian,
            np.eigh is used as eigensolver.

        Returns
        -------
        eigenvalues: np.ndarray
            eigenvalues of the matrix
        eigenvectors: np.ndarray
            eigenvectors of the matrix
        """
        mat = self.array()
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
        mat = self.array()
        assert is_hermitian(mat)
        return la.eigh(mat)

    def show(self, show=True, **kwargs):
        """Converts the `LinearOperator` to a `Matrix`-object and plots the result."""
        return self.matrix().show(show, **kwargs)


class SparseOperator(LinearOperator):
    """Sparse implementation of a `LinearOperator`."""

    __slots__ = ["rows", "cols", "data", "_csr"]

    def __init__(self, shape, dtype=None, rows=None, cols=None, data=None):
        super().__init__(shape, dtype)
        self.rows, self.cols = list(), list()
        self.data = list()
        self._csr = None
        if data is not None:
            self.set_data(rows, cols, data)

    @property
    def csr(self):
        if self._csr is None:
            arg = (self.data, (self.rows, self.cols))
            self._csr = csr_matrix(arg, self.shape, self.dtype, copy=True)
        return self._csr

    def array(self):
        return self.csr.toarray()

    def _matvec(self, x):
        """Implements matrix-vector multiplication. """
        return np.dot(self.array(), x)

    def _adjoint(self):
        """Implements the hermitian adjoint operator."""
        rows = self.rows
        cols = self.cols
        data = np.conj(self.data)
        return self.__class__(self.shape, self.dtype, rows=cols, cols=rows, data=data)

    def set_data(self, rows, cols, data):
        self._csr = None
        self.rows = list(rows)
        self.cols = list(cols)
        self.data = list(data)

    def append(self, row, col, value):
        self._csr = None
        self.rows.append(row)
        self.cols.append(col)
        self.data.append(value)

    def find(self, row, col):
        indices = np.array([self.rows, self.cols]).T
        return list(np.where(np.all(indices == np.array([row, col]), axis=1))[0])

    def collect_garbage(self):
        self.rows, self.cols = list(), list()
        self.data = list()

    def _check_indices(self, row, col):
        if row < 0:
            row = self.shape[0] + row
        if row >= self.shape[0]:
            raise IndexError(f"Row {row} out of bounds for axis of size {self.shape[0]}.")
        if col < 0:
            col = self.shape[1] + col
        if col >= self.shape[1]:
            raise IndexError(f"Column {col} out of bounds for axis of size {self.shape[1]}.")
        return row, col

    def __getitem__(self, item):
        row, col = self._check_indices(*item)
        val = 0.
        for idx in self.find(row, col):
            val += self.data[idx]
        return val

    def __setitem__(self, item, value):
        row, col = self._check_indices(*item)
        for idx in reversed(sorted(self.find(row, col))):
            del self.rows[idx]
            del self.cols[idx]
            del self.data[idx]
        self.append(row, col, value)

    def __delitem__(self, item):
        row, col = self._check_indices(*item)
        for idx in reversed(sorted(self.find(row, col))):
            del self.rows[idx]
            del self.cols[idx]
            del self.data[idx]


# =========================================================================
# Time evolution operator
# =========================================================================


class TimeEvolutionOperator(LinearOperator):

    def __init__(self, operator, t=0., t0=0., dtype=None):
        super().__init__(operator.shape, dtype=dtype)
        self.decomposition = Decomposition.decompose(operator)
        self.t = t - t0

    def reconstruct(self, xi=None, method='full'):
        return self.decomposition.reconstrunct(xi, method)

    def set_eigenbasis(self, operator):
        self.decomposition = Decomposition.decompose(operator)

    def set_time(self, t, t0=0.):
        self.t = t - t0

    def _matvec(self, x):
        rv = self.decomposition.rv
        xi = self.decomposition.xi
        # Project state into eigenbasis
        proj = np.inner(rv.T, x)
        # Evolve projected state
        proj_t = proj * np.exp(-1j * xi * self.t)
        # Reconstruct the new state in the site-basis
        return np.dot(proj_t, rv.T)

    def array(self) -> np.ndarray:
        return self.reconstruct()

    def __call__(self, t):
        self.set_time(t)
        return self

    def evolve(self, state, t):
        self.set_time(t)
        return self.matvec(state)


# =========================================================================
# Creation- and Annihilation-Operators
# =========================================================================


class CreationOperator(LinearOperator):

    def __init__(self, sector, sector_p1, pos=0, sigma=UP):
        dim_origin = sector.size
        if sigma == UP:
            dim_target = sector_p1.num_up * sector.num_dn
        else:
            dim_target = sector_p1.num_dn * sector.num_up

        super().__init__(shape=(dim_target, dim_origin), dtype=np.complex64)
        self.pos = pos
        self.sigma = sigma
        self.sector = sector
        self.sector_p1 = sector_p1

    def __repr__(self):
        name = f"{self.__class__.__name__}_{self.pos}{SPIN_CHARS[self.sigma]}"
        return f"{name}(shape: {self.shape}, dtype: {self.dtype})"

    def _build_up(self, matvec, x):
        op = 1 << self.pos
        num_dn = len(self.sector.dn_states)
        all_dn = np.arange(num_dn)
        for up_idx, up in enumerate(self.sector.up_states):
            if not (up & op):
                new = up ^ op
                idx_new = bisect_left(self.sector_p1.up_states, new)
                origins = project_up(up_idx, num_dn, all_dn)
                targets = project_up(idx_new, num_dn, all_dn)
                if isinstance(origins, int):
                    origins, targets = [origins], [targets]
                for origin, target in zip(origins, targets):
                    matvec[target] = x[origin]

    def _build_dn(self, matvec, x):
        op = 1 << self.pos
        num_dn = self.sector.num_dn
        all_up = np.arange(self.sector.num_up)
        for dn_idx, dn in enumerate(self.sector.dn_states):
            if not (dn & op):
                new = dn ^ op
                idx_new = bisect_left(self.sector_p1.dn_states, new)
                origins = project_dn(dn_idx, num_dn, all_up)
                targets = project_dn(idx_new, num_dn, all_up)
                if isinstance(origins, int):
                    origins, targets = [origins], [targets]
                for origin, target in zip(origins, targets):
                    matvec[target] = x[origin]

    def _matvec(self, x):
        newsize = self.shape[0]
        matvec = np.zeros((newsize, *x.shape[1:]), dtype=x.dtype)
        if self.sigma == UP:
            self._build_up(matvec, x)
        else:
            self._build_dn(matvec, x)
        return matvec


class AnnihilationOperator(LinearOperator):

    def __init__(self, sector, sector_m1, pos=0, sigma=UP):
        dim_origin = sector.size
        if sigma == UP:
            dim_target = sector_m1.num_up * sector.num_dn
        else:
            dim_target = sector_m1.num_dn * sector.num_up

        super().__init__(shape=(dim_target, dim_origin), dtype=np.complex64)
        self.pos = pos
        self.sigma = sigma
        self.sector = sector
        self.sector_m1 = sector_m1

    def __repr__(self):
        name = f"{self.__class__.__name__}_{self.pos}{SPIN_CHARS[self.sigma]}"
        return f"{name}(shape: {self.shape}, dtype: {self.dtype})"

    def _build_up(self, matvec, x):
        op = 1 << self.pos
        num_dn = len(self.sector.dn_states)
        all_dn = np.arange(num_dn)
        for up_idx, up in enumerate(self.sector.up_states):
            if up & op:
                new = up ^ op
                idx_new = bisect_left(self.sector_m1.up_states, new)
                origins = project_up(up_idx, num_dn, all_dn)
                targets = project_up(idx_new, num_dn, all_dn)
                if isinstance(origins, int):
                    origins, targets = [origins], [targets]
                for origin, target in zip(origins, targets):
                    matvec[target] = x[origin]

    def _build_dn(self, matvec, x):
        op = 1 << self.pos
        num_dn = self.sector.num_dn
        all_up = np.arange(self.sector.num_up)
        for dn_idx, dn in enumerate(self.sector.dn_states):
            if dn & op:
                new = dn ^ op
                idx_new = bisect_left(self.sector_m1.dn_states, new)
                origins = project_dn(dn_idx, num_dn, all_up)
                targets = project_dn(idx_new, num_dn, all_up)
                if isinstance(origins, int):
                    origins, targets = [origins], [targets]
                for origin, target in zip(origins, targets):
                    matvec[target] = x[origin]

    def _matvec(self, x):
        newsize = self.shape[0]
        matvec = np.zeros((newsize, *x.shape[1:]), dtype=x.dtype)
        if self.sigma == UP:
            self._build_up(matvec, x)
        else:
            self._build_dn(matvec, x)
        return matvec
