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
from bisect import bisect_left
import scipy.linalg as la
import scipy.sparse.linalg as sla
from typing import Union, Callable, Iterable, Sequence, Generator
from cmpy.basis import binstr, occupations, overlap, UP, SPIN_CHARS
from cmpy.matrix import Matrix, is_hermitian, Decomposition

__all__ = ["LinearOperator", "HamiltonOperator", "CreationOperator", "AnnihilationOperator",
           "project_up", "project_dn", "project_elements_up", "project_elements_dn",
           "project_interaction", "project_onsite_energy",
           "project_site_hopping", "project_hopping", "TimeEvolutionOperator"]


def project_up(up_idx: int, num_dn_states: int,
               dn_indices: Union[int, np.ndarray]) -> np.ndarray:
    """Projects a spin-up state onto the full basis(-sector).

    Parameters
    ----------
    up_idx : int
        The index of the up-state to project.
    num_dn_states : int
        The total number of spin-down states of the basis(-sector).
    dn_indices: int or (N) np.ndarray
        An array of the indices of all spin-down states in the basis(-sector).

    Returns
    -------
    projected_up : np.ndarray
        The indices of the projected up-state.

    Examples
    --------
    >>> num_dn = 4
    >>> all_dn = np.arange(4)
    >>> project_up(0, num_dn, all_dn)
    array([0, 1, 2, 3])
    >>> project_up(1, num_dn, all_dn)
    array([4, 5, 6, 7])
    """
    return np.atleast_1d(up_idx * num_dn_states + dn_indices)


def project_dn(dn_idx: int, num_dn_states: int,
               up_indices: Union[int, np.ndarray]) -> np.ndarray:
    """Projects a spin-down state onto the full basis(-sector).

    Parameters
    ----------
    dn_idx: int
        The index of the down-state to project.
    num_dn_states: int
        The total number of spin-down states of the basis(-sector).
    up_indices: int or (N) np.ndarray
        An array of the indices of all spin-up states in the basis(-sector).

    Returns
    -------
    projected_dn : np.ndarray
        The indices of the projected down-state.

    Examples
    --------
    >>> num_dn = 4
    >>> all_up = np.arange(4)
    >>> project_dn(0, num_dn, all_up)
    array([ 0,  4,  8, 12])
    >>> project_dn(1, num_dn, all_up)
    array([ 1,  5,  9, 13])
    """
    return np.atleast_1d(up_indices * num_dn_states + dn_idx)


def project_elements_up(up_idx: int, num_dn_states: int,
                        dn_indices: Union[int, np.ndarray],
                        value: Union[complex, float],
                        target: Union[int, np.ndarray] = None
                        ) -> Generator[tuple[int, int, float]]:
    """Projects a value for a spin-up state onto the elements of the full basis(-sector).

    Parameters
    ----------
    up_idx : int
        The index of the up-state to project.
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

    Examples
    --------
    >>> num_dn = 4
    >>> all_dn = np.arange(4)
    >>> np.array(list(project_elements_up(0, num_dn, all_dn, value=1)))
    array([[0, 0, 1],
           [1, 1, 1],
           [2, 2, 1],
           [3, 3, 1]])
    >>> np.array(list(project_elements_up(1, num_dn, all_dn, value=1)))
    array([[4, 4, 1],
           [5, 5, 1],
           [6, 6, 1],
           [7, 7, 1]])
    """
    if not value:
        return

    origins = project_up(up_idx, num_dn_states, dn_indices)
    if target is None:
        targets = origins
    else:
        targets = project_up(target, num_dn_states, dn_indices)

    for row, col in zip(origins, targets):
        yield row, col, value


def project_elements_dn(dn_idx: int, num_dn_states: int,
                        up_indices: Union[int, np.ndarray],
                        value: Union[complex, float],
                        target: Union[int, np.ndarray] = None
                        ) -> Generator[tuple[int, int, float]]:
    """Projects a value for a spin-down state onto the elements of the full basis(-sector).

    Parameters
    ----------
    dn_idx: int
        The index of the down-state to project.
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

    Examples
    --------
    >>> num_dn = 4
    >>> all_up = np.arange(4)
    >>> np.array(list(project_elements_dn(0, num_dn, all_up, value=1)))
    array([[ 0,  0,  1],
           [ 4,  4,  1],
           [ 8,  8,  1],
           [12, 12,  1]])
    >>> np.array(list(project_elements_dn(1, num_dn, all_up, value=1)))
    array([[ 1,  1,  1],
           [ 5,  5,  1],
           [ 9,  9,  1],
           [13, 13,  1]])
    """
    if not value:
        return

    origins = project_dn(dn_idx, num_dn_states, up_indices)
    if target is None:
        targets = origins
    else:
        targets = project_dn(target, num_dn_states, up_indices)

    for row, col in zip(origins, targets):
        yield row, col, value


# =========================================================================
# Interacting Hamiltonian projectors
# =========================================================================


def project_onsite_energy(up_states: Sequence[int], dn_states: Sequence[int],
                          eps: float) -> Generator[tuple[int, int, float]]:
    """Projects the on-site energy of a many-body Hamiltonian onto full basis(-sector).

    Parameters
    ----------
    up_states : array_like
        An array of all spin-up states in the basis(-sector).
    dn_states : array_like
        An array of all spin-down states in the basis(-sector).
    eps : float
        The on-site energy.

    Yields
    ------
    row: int
        The row-index of the on-site energy.
    col: int
        The column-index of the on-site energy.
    value: float
        The on-site energy.
    """
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


def project_interaction(up_states: Sequence[int], dn_states: Sequence[int],
                        u: float) -> Generator[tuple[int, int, float]]:
    """Projects the on-site interaction of a many-body Hamiltonian onto full basis(-sector).

    Parameters
    ----------
    up_states : array_like
        An array of all spin-up states in the basis(-sector).
    dn_states : array_like
        An array of all spin-down states in the basis(-sector).
    u : float
        The on-site interaction.

    Yields
    ------
    row: int
        The row-index of the on-site interaction.
    col: int
        The column-index of the on-site interaction.
    value: float
        The on-site interaction.
    """
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


def project_site_hopping(up_states: Sequence[int], dn_states: Sequence[int],
                         num_sites: int, hopping: Union[Callable, Iterable, float],
                         pos: int) -> Generator[tuple[int, int, float]]:
    """Projects the hopping of a single site of a many-body Hamiltonian onto full basis(-sector).

    Parameters
    ----------
    up_states : array_like
        An array of all spin-up states in the basis(-sector).
    dn_states : array_like
        An array of all spin-down states in the basis(-sector).
    num_sites : int
        The number of sites in the model.
    hopping : callable or array_like
        An iterable or callable defining the hopping energy. If a callable is used
        the two positions of the hopping elements are passed to the method. Otherwise,
        the positions are used as indices.
    pos : int
        The index of the position considered in the hopping processes.

    Yields
    ------
    row: int
        The row-index of the hopping element.
    col: int
        The column-index of the hopping element.
    value: float
        The hopping energy.
    """
    num_dn = len(dn_states)
    all_up, all_dn = np.arange(len(up_states)), np.arange(num_dn)

    for up_idx, target, amp in _compute_hopping(num_sites, up_states, pos, hopping):
        yield from project_elements_up(up_idx, num_dn, all_dn, amp, target=target)

    for dn_idx, target, amp in _compute_hopping(num_sites, dn_states, pos, hopping):
        yield from project_elements_dn(dn_idx, num_dn, all_up, amp, target=target)


def project_hopping(up_states: Sequence[int], dn_states: Sequence[int],
                    num_sites: int, hopping: Union[Callable, Iterable, float]
                    ) -> Generator[tuple[int, int, float]]:
    """Projects the hopping of all sites of a many-body Hamiltonian onto full basis(-sector).

    Parameters
    ----------
    up_states : array_like
        An array of all spin-up states in the basis(-sector).
    dn_states : array_like
        An array of all spin-down states in the basis(-sector).
    num_sites : int
        The number of sites in the model.
    hopping : callable or array_like
        An iterable or callable defining the hopping energy. If a callable is used
        the two positions of the hopping elements are passed to the method. Otherwise,
        the positions are used as indices.

    See Also
    --------
    project_site_hopping

    Yields
    ------
    row: int
        The row-index of the hopping element.
    col: int
        The column-index of the hopping element.
    value: float
        The hopping energy.
    """
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
        """Calculate eigenvalues and -vectors of the hermitian matrix.

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

    def eigsh(self, k=6, which="SA", **kwargs):
        return sla.eigsh(self, k=k, which=which, **kwargs)  # noqa

    def show(self, show=True, **kwargs):
        """Converts the `LinearOperator` to a `Matrix`-object and plots the result."""
        return self.matrix().show(show, **kwargs)


# =========================================================================
# Hamilton-operator
# =========================================================================


class HamiltonOperator(LinearOperator):

    def __init__(self, size, data, indices, dtype=None):
        data = np.asarray(data)
        indices = np.asarray(indices)
        if dtype is None:
            dtype = data.dtype
        super().__init__((size, size), dtype=dtype)
        self.data = data
        self.indices = indices.T

    def _matvec(self, x):
        matvec = np.zeros_like(x)
        for (row, col), val in zip(self.indices, self.data):
            matvec[col] += val * x[row]
        return matvec

    def _adjoint(self):
        return self

    def trace(self):
        # Todo: Sparse trace method
        return np.trace(self.array())

    def __mul__(self, x):
        """Ensure methods in result."""
        scaled = super().__mul__(x)
        scaled.trace = lambda: x * self.trace()
        scaled.array = lambda: x * self.array()
        scaled.matrix = lambda: x * self.matrix()
        return scaled

    def __rmul__(self, x):
        """Ensure methods in result."""
        scaled = super().__rmul__(x)
        scaled.trace = lambda: x * self.trace()
        scaled.array = lambda: x * self.array()
        scaled.matrix = lambda: x * self.matrix()
        return scaled


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
        matvec = np.zeros((self.shape[0], *x.shape[1:]), dtype=x.dtype)
        if self.sigma == UP:
            self._build_up(matvec, x)
        else:
            self._build_dn(matvec, x)
        return matvec

    def _adjoint(self):
        return AnnihilationOperator(self.sector_p1, self.sector, self.pos, self.sigma)


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
        matvec = np.zeros((self.shape[0], *x.shape[1:]), dtype=x.dtype)
        if self.sigma == UP:
            self._build_up(matvec, x)
        else:
            self._build_dn(matvec, x)
        return matvec

    def _adjoint(self):
        return CreationOperator(self.sector_m1, self.sector, self.pos, self.sigma)


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
