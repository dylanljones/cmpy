# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2022, Dylan Jones

"""This module contains tools for working with linear operators in sparse format."""

import abc
import warnings
import numpy as np
from bisect import bisect_left
import scipy.sparse.linalg as sla
from typing import Union, Callable, Iterable, Sequence
from .basis import bit_count, occupations, overlap, UP, SPIN_CHARS
from .matrix import Decomposition

__all__ = [
    "LinearOperator",
    "HamiltonOperator",
    "CreationOperator",
    "AnnihilationOperator",
    "project_up",
    "project_dn",
    "project_elements_up",
    "project_elements_dn",
    "project_hubbard_inter",
    "project_onsite_energy",
    "project_site_hopping",
    "project_hopping",
    "TimeEvolutionOperator",
]


def project_up(
    up_idx: int, num_dn_states: int, dn_indices: Union[int, np.ndarray]
) -> np.ndarray:
    """Projects a spin-up state onto the full basis(-sector).

    Parameters
    ----------
    up_idx : int
        The index of the up-state to project.
    num_dn_states : int
        The number N of spin-down states of the basis(-sector).
    dn_indices : int or (N, ) np.ndarray
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


def project_dn(
    dn_idx: int, num_dn_states: int, up_indices: Union[int, np.ndarray]
) -> np.ndarray:
    """Projects a spin-down state onto the full basis(-sector).

    Parameters
    ----------
    dn_idx : int
        The index of the down-state to project.
    num_dn_states : int
        The number N of spin-down states of the basis(-sector).
    up_indices : int or (N, ) np.ndarray
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


def project_elements_up(
    up_idx: int,
    num_dn_states: int,
    dn_indices: Union[int, np.ndarray],
    value: Union[complex, float, np.ndarray],
    target: Union[int, np.ndarray] = None,
):
    """Projects a value for a spin-up state onto the full basis(-sector).

    Parameters
    ----------
    up_idx : int
        The index of the up-state to project.
    num_dn_states : int
        The total number of spin-down states of the basis(-sector).
    dn_indices : int or np.ndarray
        An array of the indices of all spin-down states in the basis(-sector).
    value : float or complex
        The value to project.
    target : int or np.ndarray, optional
        The target index/indices for the projection. This is only needed
        for non-diagonal elements.

    Yields
    -------
    row : int
        The row-index of the element.
    col : int
        The column-index of the element.
    value : float or complex
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

    >>> np.array(list(project_elements_up(0, num_dn, all_dn, value=1, target=1)))
    array([[0, 4, 1],
           [1, 5, 1],
           [2, 6, 1],
           [3, 7, 1]])

    >>> np.array(list(project_elements_up(1, num_dn, all_dn, value=1)))
    array([[4, 4, 1],
           [5, 5, 1],
           [6, 6, 1],
           [7, 7, 1]])

    >>> np.array(list(project_elements_up(1, num_dn, all_dn, value=1, target=2)))
    array([[ 4,  8,  1],
           [ 5,  9,  1],
           [ 6, 10,  1],
           [ 7, 11,  1]])
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


def project_elements_dn(
    dn_idx: int,
    num_dn_states: int,
    up_indices: Union[int, np.ndarray],
    value: Union[complex, float, np.ndarray],
    target: Union[int, np.ndarray] = None,
):
    """Projects a value for a spin-down state onto the full basis(-sector).

    Parameters
    ----------
    dn_idx : int
        The index of the down-state to project.
    num_dn_states : int
        The total number of spin-down states of the basis(-sector).
    up_indices : int or np.ndarray
        An array of the indices of all spin-up states in the basis(-sector).
    value : float or complex
        The value to project.
    target : int or np.ndarray, optional
        The target index/indices for the projection. This is only needed
        for non-diagonal elements.

    Yields
    -------
    row : int
        The row-index of the element.
    col : int
        The column-index of the element.
    value : float or complex
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

    >>> np.array(list(project_elements_dn(0, num_dn, all_up, value=1, target=1)))
    array([[ 0,  1,  1],
           [ 4,  5,  1],
           [ 8,  9,  1],
           [12, 13,  1]])

    >>> np.array(list(project_elements_dn(1, num_dn, all_up, value=1)))
    array([[ 1,  1,  1],
           [ 5,  5,  1],
           [ 9,  9,  1],
           [13, 13,  1]])

    >>> np.array(list(project_elements_dn(1, num_dn, all_up, value=1, target=2)))
    array([[ 1,  2,  1],
           [ 5,  6,  1],
           [ 9, 10,  1],
           [13, 14,  1]])

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


# -- Interacting Hamiltonian projectors ------------------------------------------------


def project_onsite_energy(
    up_states: Sequence[int], dn_states: Sequence[int], eps: Sequence[float]
):
    """Projects the on-site energy of a many-body Hamiltonian onto full basis(-sector).

    Parameters
    ----------
    up_states : array_like
        An array of all spin-up states in the basis(-sector).
    dn_states : array_like
        An array of all spin-down states in the basis(-sector).
    eps : array_like
        The on-site energy.

    Yields
    ------
    row : int
        The row-index of the on-site energy.
    col : int
        The column-index of the on-site energy.
    value : float
        The on-site energy.

    Examples
    --------
    >>> from cmpy import Basis
    >>> basis = Basis(num_sites=2)
    >>> sector = basis.get_sector(n_up=1, n_dn=1)
    >>> up_states, dn_states = sector.up_states, sector.dn_states
    >>> energies = [1.0, 2.0]
    >>> ham = np.zeros((sector.size, sector.size))
    >>> for i, j, val in project_onsite_energy(up_states, dn_states, energies):
    ...     ham[i, j] += val
    >>> ham
    array([[2., 0., 0., 0.],
           [0., 3., 0., 0.],
           [0., 0., 3., 0.],
           [0., 0., 0., 4.]])

    >>> from cmpy import matshow
    >>> matshow(ham, ticklabels=sector.state_labels(), values=True)

    """
    num_dn = len(dn_states)
    all_up, all_dn = np.arange(len(up_states)), np.arange(num_dn)

    for up_idx, up in enumerate(up_states):
        weights = occupations(up)
        energy = np.sum(eps[: weights.size] * weights)
        yield from project_elements_up(up_idx, num_dn, all_dn, energy)

    for dn_idx, dn in enumerate(dn_states):
        weights = occupations(dn)
        energy = np.sum(eps[: weights.size] * weights)
        yield from project_elements_dn(dn_idx, num_dn, all_up, energy)


def project_hubbard_inter(
    up_states: Sequence[int], dn_states: Sequence[int], u: Sequence[float]
):
    """Projects the on-site interaction of a many-body Hamiltonian onto the full basis.

    Parameters
    ----------
    up_states : array_like
        An array of all spin-up states in the basis(-sector).
    dn_states : array_like
        An array of all spin-down states in the basis(-sector).
    u : array_like
        The on-site interaction.

    Yields
    ------
    row : int
        The row-index of the on-site interaction.
    col : int
        The column-index of the on-site interaction.
    value : float
        The on-site interaction.

    Examples
    --------
    >>> from cmpy import Basis
    >>> basis = Basis(num_sites=2)
    >>> sector = basis.get_sector(n_up=1, n_dn=1)
    >>> up_states, dn_states = sector.up_states, sector.dn_states
    >>> inter = [1.0, 2.0]
    >>> ham = np.zeros((sector.size, sector.size))
    >>> for i, j, val in project_hubbard_inter(up_states, dn_states, inter):
    ...     ham[i, j] += val
    >>> ham
    array([[1., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 2.]])

    >>> from cmpy import matshow
    >>> matshow(ham, ticklabels=sector.state_labels(), values=True)

    """
    num_dn = len(dn_states)
    for up_idx, up in enumerate(up_states):
        for dn_idx, dn in enumerate(dn_states):
            weights = overlap(up, dn)
            interaction = np.sum(u[: weights.size] * weights)
            yield from project_elements_up(up_idx, num_dn, dn_idx, interaction)


def _hopping_sign(initial_state, site1, site2):
    mask = int(sum(1 << x for x in range(site1 + 1, site2)))
    jump_overs = bit_count(initial_state & mask)
    sign = (-1) ** jump_overs
    return sign


def _compute_hopping_term(states, site1, site2, hop):
    assert site1 < site2

    for i, ini in enumerate(states):
        op1 = 1 << site1  # Selects bit with index `site1`
        occ1 = ini & op1  # Value of bit of state with index `site1`
        tmp = ini ^ op1  # Annihilate/create electron at `site1`

        op2 = 1 << site2  # Selects bit with index `site2`
        occ2 = ini & op2  # Value of bit of state with index `site2`
        new = tmp ^ op2  # Create/annihilate electron at `site1`

        # ToDo: Account for hop-overs of other spin flavour
        if occ1 and not occ2:
            # Hopping from `site1` to `site2` possible
            sign = _hopping_sign(ini, site1, site2)
            j = bisect_left(states, new)
            yield i, j, sign * hop
        elif occ2 and not occ1:
            # Hopping from `site2` to `site1` possible
            sign = _hopping_sign(ini, site1, site2)
            j = bisect_left(states, new)
            yield i, j, sign * hop


def project_hopping(
    up_states: Sequence[int],
    dn_states: Sequence[int],
    site1: int,
    site2: int,
    hop: float,
):
    """Projects the hopping between two sites onto full basis.

    Parameters
    ----------
    up_states : array_like
        An array of all spin-up states in the basis(-sector).
    dn_states : array_like
        An array of all spin-down states in the basis(-sector).
    site1 : int
        The first site of the hopping pair. This has to be the lower index of the two
        sites.
    site2 : int
        The second site of the hopping pair. This has to be the larger index of the two
        sites.
    hop : float, optional
        The hopping energy between the two sites.

    Yields
    ------
    row : int
        The row-index of the hopping element.
    col : int
        The column-index of the hopping element.
    value : float
        The hopping energy.

    Examples
    --------
    >>> from cmpy import Basis
    >>> basis = Basis(num_sites=2)
    >>> sector = basis.get_sector(n_up=1, n_dn=1)
    >>> up, dn = sector.up_states, sector.dn_states
    >>> ham = np.zeros((sector.size, sector.size))
    >>> for i, j, val in project_hopping(up, dn, site1=0, site2=1, hop=1.0):
    ...     ham[i, j] += val
    >>> ham
    array([[0., 1., 1., 0.],
           [1., 0., 0., 1.],
           [1., 0., 0., 1.],
           [0., 1., 1., 0.]])

    >>> from cmpy import matshow
    >>> matshow(ham, ticklabels=sector.state_labels(), values=True)

    """
    if site1 > site2:
        raise ValueError("The first site index must be smaller than the second one!")

    num_dn = len(dn_states)
    all_up, all_dn = np.arange(len(up_states)), np.arange(num_dn)

    for idx, target, amp in _compute_hopping_term(up_states, site1, site2, hop):
        yield from project_elements_up(idx, num_dn, all_dn, amp, target=target)

    for idx, target, amp in _compute_hopping_term(dn_states, site1, site2, hop):
        yield from project_elements_dn(idx, num_dn, all_up, amp, target=target)


def _hopping_candidates(num_sites, state, pos):
    results = []
    op = 1 << pos
    occ = state & op
    sign_to = 1
    sign_from = 1
    tmp = state ^ op  # Annihilate or create electron at `pos`
    for pos2 in range(num_sites):
        if pos >= pos2:
            continue
        op2 = 1 << pos2
        occ2 = state & op2

        # Hopping from `pos` to `pos2` possible
        if occ and not occ2:
            new = tmp ^ op2
            results.append((pos2, new, sign_to))
        else:  # state filled, no hopping but sign change
            sign_to *= -1

        # Hopping from `pos2` to `pos` possible
        if not occ and occ2:
            new = tmp ^ op2
            results.append((pos2, new, sign_from))
            sign_from *= -1  # if this site is jumped over sign changes
    return results


def _compute_hopping(num_sites, states, pos, hopping):
    for i, state in enumerate(states):
        for pos2, new, sign in _hopping_candidates(num_sites, state, pos):
            try:
                t = hopping(pos, pos2)
            except TypeError:
                t = hopping
            if t:
                j = bisect_left(states, new)
                value = sign * t
                yield i, j, value


def project_site_hopping(
    up_states: Sequence[int],
    dn_states: Sequence[int],
    num_sites: int,
    hopping: Union[Callable, Iterable, float],
    pos: int,
):
    """Projects the hopping of a single site of a many-body Hamiltonian onto full basis.

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
    warnings.warn(
        "This method is deprecated! Use 'project_hopping' istead!", DeprecationWarning
    )
    num_dn = len(dn_states)
    all_up, all_dn = np.arange(len(up_states)), np.arange(num_dn)

    for up_idx, target, amp in _compute_hopping(num_sites, up_states, pos, hopping):
        yield from project_elements_up(up_idx, num_dn, all_dn, amp, target=target)

    for dn_idx, target, amp in _compute_hopping(num_sites, dn_states, pos, hopping):
        yield from project_elements_dn(dn_idx, num_dn, all_up, amp, target=target)


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
        Returns the Hermitian adjoint of self, aka the Hermitian conjugate or Hermitian
        transpose. For a complex matrix, the Hermitian adjoint is equal to the conjugate
        transpose. Can be abbreviated self.H instead of self.adjoint(). As with
        _matvec and _matmat, implementing either _rmatvec or _adjoint implements the
        other automatically. Implementing _adjoint is preferable!

    _trace(): Trace of operator.
        Computes the trace of the operator using the a dense array.
        Implementing _trace with a more sophisticated method is preferable!
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

    def _trace(self) -> float:
        """Naive implementation of trace. Override for more efficient calculation."""
        x = np.eye(self.shape[1], dtype=self.dtype)
        return float(np.trace(self.matmat(x)))

    def trace(self) -> float:
        """Computes the trace of the ``LinearOperator``."""
        return self._trace()

    def __mul__(self, x):
        """Ensure methods in result."""
        scaled = super().__mul__(x)
        try:
            scaled.trace = lambda: x * self.trace()
            scaled.array = lambda: x * self.array()
        except AttributeError:
            pass
        return scaled

    def __rmul__(self, x):
        """Ensure methods in result."""
        scaled = super().__rmul__(x)
        try:
            scaled.trace = lambda: x * self.trace()
            scaled.array = lambda: x * self.array()
        except AttributeError:
            pass
        return scaled


# -- Hamilton operator -----------------------------------------------------------------


class HamiltonOperator(LinearOperator):
    """Hamiltonian as LinearOperator."""

    def __init__(self, size, data, indices, dtype=None):
        data = np.asarray(data)
        indices = np.asarray(indices)
        if dtype is None:
            dtype = data.dtype
        super().__init__((size, size), dtype=dtype)
        self.data = data
        self.indices = indices.T

    def _matvec(self, x) -> np.ndarray:
        matvec = np.zeros_like(x)
        for (row, col), val in zip(self.indices, self.data):
            matvec[col] += val * x[row]
        return matvec

    def _adjoint(self) -> "HamiltonOperator":
        """Hamiltonian is hermitian."""
        return self

    def _trace(self) -> float:
        """More efficient trace."""
        # Check elements where the row equals the column
        indices = np.where(self.indices[:, 0] == self.indices[:, 1])[0]
        # Return sum of diagonal elements
        return float(np.sum(self.data[indices]))


# -- Creation- and Annihilation-Operators ----------------------------------------------


class CreationOperator(LinearOperator):
    """Fermionic creation operator as LinearOperator."""

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
    """Fermionic annihilation operator as LinearOperator."""

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


# -- Time evolution operator -----------------------------------------------------------


class TimeEvolutionOperator(LinearOperator):
    """Time evolution operator as LinearOperator."""

    def __init__(self, operator, t=0.0, t0=0.0, dtype=None):
        super().__init__(operator.shape, dtype=dtype)
        self.decomposition = Decomposition.decompose(operator)
        self.t = t - t0

    def reconstruct(self, xi=None, method="full"):
        return self.decomposition.reconstruct(xi, method)

    def set_eigenbasis(self, operator):
        self.decomposition = Decomposition.decompose(operator)

    def set_time(self, t, t0=0.0):
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
