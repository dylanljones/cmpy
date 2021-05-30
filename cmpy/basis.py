# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Tools for handling and representing (fermionic) Fock basis-states."""

import numpy as np
from itertools import product, permutations
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Union, Iterable

__all__ = ["UP", "DN", "SPIN_CHARS", "state_label", "binstr", "binarr",
           "binidx", "overlap", "occupations", "create", "annihilate",
           "SpinState", "State", "Sector", "Basis"]

_BITORDER = +1      # endianess of binary strings (index 0 is of the rhs)
_ARRORDER = -1      # endianess of binary arrays (index 0 is of the lhs)
_LABELORDERER = +1  # endianess of state labels (index 0 is of the rhs)

UP, DN = 1, 2   # constants for up/down

EMPTY_CHAR = "."
UP_CHAR = "↑"
DN_CHAR = "↓"
UD_CHAR = "⇅"  # "d"
SPIN_CHARS = {0: EMPTY_CHAR, UP: UP_CHAR, DN: DN_CHAR, 3: UD_CHAR}


def state_label(up_num: int, dn_num: int, digits: Optional[int] = None) -> str:
    """Creates a string representing the spin-up and spin-down basis state.

    Parameters
    ----------
    up_num : int
        The number representing the binary spin-up basis state.
    dn_num : int
        The number representing the binary spin-dn basis state.
    digits : int, optional
        Minimum number of digits used. The default is ``None`` (no padding).

    Returns
    -------
    label : str
    """
    digits = digits if digits is not None else 0
    min_digits = max(len(bin(up_num)[2:]), len(bin(dn_num)[2:]))
    num_chars = max(min_digits, digits)
    chars = list()
    for i in range(num_chars):
        u = up_num >> i & 1
        d = dn_num >> i & 1
        chars.append(SPIN_CHARS[u + (d << 1)])
    label = "".join(chars[::_LABELORDERER])
    return label


# =========================================================================
# Binary methods
# =========================================================================


def binstr(num: int, width: Optional[int] = 0) -> str:
    """Returns the binary representation of an integer as string.

    Parameters
    ----------
    num : int
        The number to represent as binary string.
    width : int, optional
        Minimum number of digits used. The default is ``None`` (no padding).

    Returns
    -------
    binstr : str
        The binary string representing the given number.

    Examples
    --------
    >>> binstr(0)
    '0'
    >>> binstr(3)
    '11'
    >>> binstr(4)
    '100'
    >>> binstr(0, width=4)
    '0000'
    >>> binstr(3, width=4)
    '0011'
    >>> binstr(4, width=4)
    '0100'
    """
    width = width if width is not None else 0
    return f"{num:0{width}b}"[::_BITORDER]


def binarr(num: int, width: Optional[int] = None,
           dtype: Optional[Union[int, str]] = None) -> np.ndarray:
    """Returns the bits of an integer as a binary array.

    Parameters
    ----------
    num : int or Spinstate
        The number representing the binary state.
    width : int, optional
        Minimum number of digits used. The default is ``None`` (no padding).
    dtype : int or str, optional
        An optional datatype-parameter. The default is ``None``.

    Returns
    -------
    binarr : np.ndarray
        The binary array representing the given number.

    Examples
    --------
    >>> binarr(0)           # binary:    0
    array([0], dtype=int64)
    >>> binarr(3)           # binary:   11
    array([1, 1], dtype=int64)
    >>> binarr(4)           # binary:  100
    array([0, 0, 1], dtype=int64)
    >>> binarr(0, width=4)  # binary: 0000
    array([0, 0, 0, 0], dtype=int64)
    >>> binarr(3, width=4)  # binary: 0011
    array([1, 1, 0, 0], dtype=int64)
    >>> binarr(4, width=4)  # binary: 0100
    array([0, 0, 1, 0], dtype=int64)
    """
    width = width if width is not None else 0
    dtype = dtype or np.int64
    return np.fromiter(f"{num:0{width}b}"[::_ARRORDER], dtype=dtype)


def binidx(num, width: Optional[int] = None) -> Iterable[int]:
    """Returns the indices of bits with the value ``1``.

    Parameters
    ----------
    num : int or Spinstate
        The number representing the binary state.
    width : int, optional
        Minimum number of digits used. The default is ``None`` (no padding).

    Returns
    -------
    binidx : list
        List of all indices of the bits set to ``1``.

    Examples
    --------
    >>> binidx(0)           # binary:    0
    []
    >>> binidx(3)           # binary:   11
    [0, 1]
    >>> binidx(4)           # binary:  100
    [2]
    >>> binidx(0, width=4)  # binary: 0000
    []
    >>> binidx(3, width=4)  # binary: 0011
    [0, 1]
    >>> binidx(4, width=4)  # binary: 0100
    [2]
    """
    width = width if width is not None else 0
    b = f"{num:0{width}b}"
    return list(sorted(i for i, char in enumerate(b[::_ARRORDER]) if char == "1"))


def overlap(num1: int, num2: int, width: Optional[int] = None,
            dtype: Optional[Union[int, str]] = None) -> np.ndarray:
    """Computes the overlap of two integers and returns the results as a binary array.

    Parameters
    ----------
    num1 : int
        The integer representing the first binary state.
    num2 : int
        The integer representing the second binary state.
    width : int, optional
        Minimum number of digits used. The default is ``None`` (no padding).
    dtype : int or str, optional
        An optional datatype-parameter. The default is ``None``.

    Returns
    -------
    binarr : np.ndarray
        Binary array containing the overlaps.

    Examples
    --------
    >>> overlap(0, 0)           # binary:    0,    0
    array([0], dtype=int64)
    >>> overlap(3, 1)           # binary:   11,    1
    array([1], dtype=int64)
    >>> overlap(0, 0, width=4)  # binary: 0000, 0000
    array([0, 0, 0, 0], dtype=int64)
    >>> overlap(3, 1, width=4)  # binary: 0011, 0001
    array([1, 0, 0, 0], dtype=int64)
    """
    return binarr(num1 & num2, width, dtype)


def occupations(num: int, width: Optional[int] = None,
                dtype: Optional[Union[int, str]] = None) -> np.ndarray:
    """Returns the site occupations of a state as a binary array.

    Parameters
    ----------
    num : int
        The number representing the binary state.
    width : int, optional
        Minimum number of digits used. The default is ``None`` (no padding).
    dtype : int or str, optional
        An optional datatype-parameter. The default is ``None``.

    Returns
    -------
    binarr : np.ndarray
        The binary array representing the bits set to ``1``.
    """
    return binarr(num, width, dtype)


def create(num: int, pos: int) -> Union[int, None]:
    """Creates a particle at ``pos`` if possible and returns the new state.

    Parameters
    ----------
    num : int or Spinstate
        The number representing the binary state.
    pos : int
        The index of the state element.

    Returns
    -------
    new : int or None
        The newly created state. If it is not possible to create the state ``None`` is returned.

    Examples
    --------
    >>> new = create(0, pos=0)  # binary:  0000
    >>> binstr(new, width=4)
    '0001'
    >>> new = create(1, pos=0)  # binary:  0001
    >>> new is None
    True
    >>> new = create(1, pos=1)  # binary:  0001
    >>> binstr(new, width=4)
    '0011'
    """
    op = 1 << pos
    if not op & num:
        return num ^ op
    return None


def annihilate(num: int, pos: int) -> Union[int, None]:
    """Annihilates a particle at ``pos`` if possible and returns the new state.

    Parameters
    ----------
    num : int or Spinstate
        The number representing the binary state.
    pos : int
        The index of the state element.

    Returns
    -------
    new : int or None
        The newly created state. If it is not possible to annihilate the state ``None`` is returned.

    Examples
    --------
    >>> new = annihilate(0, pos=0)  # binary:  0000
    >>> new is None
    True
    >>> new = annihilate(1, pos=0)  # binary:  0001
    >>> binstr(new, width=4)
    '0000'
    >>> new = annihilate(3, pos=1)  # binary:  0011
    >>> binstr(new, width=4)
    '0001'
    """
    op = 1 << pos
    if op & num:
        return num ^ op
    return None


# =========================================================================
# State wrapper and container
# =========================================================================


class SpinState(int):

    @property
    def n(self) -> int:
        """Total occupation of the state"""
        return bin(self).count("1")

    def binstr(self, width: Optional[int] = None) -> str:
        """Returns the binary representation of the state"""
        return binstr(self, width)

    def binarr(self, width: Optional[int] = None,
               dtype: Optional[Union[int, str]] = None) -> np.ndarray:
        """Returns the bits of the integer as a binary array."""
        return binarr(self, width, dtype)

    def occ(self, pos: int) -> int:
        """Returns the occupation at index `pos`."""
        return self & (1 << pos)

    def occupations(self, dtype: Optional[Union[int, str]] = None) -> np.ndarray:
        """Returns the site occupations of a state as a binary array."""
        return binarr(self, dtype=dtype)

    def overlap(self, other: Union[int, 'SpinState'],
                dtype: Optional[Union[int, str]] = None) -> np.ndarray:
        """Computes the overlap with another state and returns the results as a binary array."""
        return overlap(self, other, dtype=dtype)

    def create(self, pos: int) -> Union['SpinState', None]:
        """Creates a particle at `pos` if possible and returns the new state."""
        num = create(self, pos)
        if num is None:
            return None
        return self.__class__(num)

    def annihilate(self, pos: int) -> Union['SpinState', None]:
        """Annihilates a particle at `pos` if possible and returns the new state."""
        num = annihilate(self, pos)
        if num is None:
            return None
        return self.__class__(num)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.binstr()})"

    def __str__(self) -> str:
        return self.binstr()


@dataclass
class State:
    """Container class for a state consisting of a spin-up and spin-down basis state."""

    __slots__ = ['up', 'dn', 'num_sites']

    def __init__(self, up: Union[int, SpinState], dn: Union[int, SpinState],
                 num_sites: Optional[int] = None):
        self.up = SpinState(up)
        self.dn = SpinState(dn)
        self.num_sites = num_sites

    def label(self, width: Optional[int] = None) -> str:
        return state_label(self.up, self.dn, width if width is not None else self.num_sites)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.up}, {self.dn})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.label()}"


# =========================================================================
# Main basis objects
# =========================================================================


def upper_sector(n_up, n_dn, sigma, num_sites):
    if sigma == UP and n_up < num_sites:
        return n_up + 1, n_dn
    elif sigma == DN and n_dn < num_sites:
        return n_up, n_dn + 1
    return None


def lower_sector(n_up, n_dn, sigma):
    if sigma == UP and n_up > 0:
        return n_up - 1, n_dn
    elif sigma == DN and n_dn > 0:
        return n_up, n_dn - 1
    return None


class Sector:
    """Container class for the spin-up and spin-down states of a sector of the full basis."""

    __slots__ = ['num_sites', 'n_up', 'n_dn', 'up_states', 'dn_states']

    def __init__(self, up_states, dn_states, n_up=None, n_dn=None, num_sites=0):
        self.num_sites = num_sites
        self.n_up = n_up
        self.n_dn = n_dn
        self.up_states = up_states
        self.dn_states = dn_states

    @property
    def states(self):
        for up, dn in product(self.up_states, self.dn_states):
            yield State(up, dn, num_sites=self.num_sites)

    @property
    def size(self):
        return len(self.up_states) * len(self.dn_states)

    @property
    def num_up(self):
        return len(self.up_states)

    @property
    def num_dn(self):
        return len(self.dn_states)

    @property
    def filling(self):
        return self.n_up, self.n_dn

    def state_labels(self):
        return [s.label(self.num_sites) for s in self.states]

    def __repr__(self):
        filling = f"[{self.n_up}, {self.n_dn}]"
        return f"{self.__class__.__name__}(size: {self.size}, " \
               f"num_sites: {self.num_sites}, filling: {filling})"

    def __str__(self):
        return f"{self.__class__.__name__}({self.n_up}, {self.n_dn}, size: {self.size})"

    def __iter__(self):
        return iter(self.states)


class Basis:
    """Container class all basis states the full Hilbert space of a model."""

    __slots__ = ["size", "num_sites", "num_spinstates", "sectors", "fillings"]

    def __init__(self, num_sites: Optional[int] = 0, init_sectors: Optional[bool] = False):
        self.size = 0
        self.num_sites = 0
        self.num_spinstates = 0
        self.sectors = defaultdict(list)
        self.fillings = list(range(self.num_sites + 1))
        self.init(num_sites, init_sectors)

    def init(self, num_sites: int, init_sectors: Optional[bool] = False):
        self.num_sites = num_sites
        self.num_spinstates = 2 ** num_sites
        self.size = self.num_spinstates ** 2
        self.sectors = defaultdict(list)
        self.fillings = list(range(self.num_sites + 1))
        if init_sectors:
            for state in range(2 ** num_sites):
                n = f"{state:b}".count("1")
                self.sectors[n].append(state)

    def generate_states(self, n: int = None) -> Union[list, np.ndarray]:
        total = self.num_sites
        if n is None:
            return list(range(2 ** total))
        elif n == 0:
            return [0]
        elif n == 1:
            return list(2 ** b for b in range(total))
        bitvals = ["0" for _ in range(total - n)]
        bitvals += ["1" for _ in range(n)]
        states = set(int("".join(bits), 2) for bits in permutations(bitvals))
        return np.asarray(sorted(states))

    def get_states(self, n: int = None) -> list[int]:
        if n in self.sectors:
            # Get cached spin-sector states
            states = self.sectors.get(n, list(range(self.num_spinstates)))
        else:
            # Compute new spin-sector states and store them
            states = self.generate_states(n)
            self.sectors[n] = states
        return states

    def get_sector(self, n_up: int = None, n_dn: int = None) -> Sector:
        up_states = self.get_states(n_up)
        dn_states = self.get_states(n_dn)
        return Sector(up_states, dn_states, n_up, n_dn, self.num_sites)

    def iter_fillings(self):
        return product(self.fillings, repeat=2)

    def iter_sectors(self):
        for n_up, n_dn in product(self.fillings, repeat=2):
            yield self.get_sector(n_up, n_dn)

    def keys(self):
        return self.fillings

    def check(self, n_up, n_dn):
        fillings = self.fillings
        return n_up in fillings and n_dn in fillings

    def upper_sector(self, n_up, n_dn, sigma):
        fillings = upper_sector(n_up, n_dn, sigma, self.num_sites)
        if fillings is not None:
            return self.get_sector(*fillings)
        return None

    def lower_sector(self, n_up, n_dn, sigma):
        fillings = lower_sector(n_up, n_dn, sigma)
        if fillings is not None:
            return self.get_sector(*fillings)
        return None

    def __getitem__(self, item):
        if hasattr(item, "__len__"):
            return self.get_sector(*item)
        return self.get_states(item)

    def __repr__(self):
        return f"{self.__class__.__name__}(size: {self.size}, num_sites: {self.num_sites}, " \
               f"fillings: {self.fillings})"
