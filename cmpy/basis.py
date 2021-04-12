# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2020, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import numpy as np
from itertools import product, permutations
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Union, Iterable

__all__ = ["UP", "DN", "SPIN_CHARS", "state_label", "binstr", "binarr",
           "binidx", "overlap", "occupations", "create", "annihilate",
           "Binary", "SpinState", "State", "Sector", "Basis"]

_ARRORDER = -1

UP, DN = 1, 2

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
    label = "".join(chars)[::-1]
    return label


# =========================================================================
# Binary methods
# =========================================================================


def binstr(num: int, width: Optional[int] = 0) -> str:
    """Returns the binary representation of an integer.

    Parameters
    ----------
    num : int
        The number to convert.
    width : int, optional
        Minimum number of digits used. The default is ``None`` (no padding).

    Returns
    -------
    binstr : str
    """
    width = width if width is not None else 0
    return f"{num:0{width}b}"


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
    """
    width = width if width is not None else 0
    dtype = dtype or np.int
    return np.fromiter(f"{num:0{width}b}"[::_ARRORDER], dtype=dtype)


def binidx(num, width: Optional[int] = None) -> Iterable[int]:
    """Returns the indices of bits with the value 1.

    Parameters
    ----------
    num : int or Spinstate
        The number representing the binary state.
    width : int, optional
        Minimum number of digits used. The default is ``None`` (no padding).

    Returns
    -------
    binidx : list
    """
    b = binstr(num, width)
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
    """
    op = 1 << pos
    if op & num:
        return num ^ op
    return None


class Binary(int):

    def __init__(self, *args, width=None, **kwargs):  # noqa
        super().__init__()

    def __new__(cls, *args, width: Optional[int] = None, **kwargs):
        self = int.__new__(cls, *args)  # noqa
        self.width = width
        return self

    @property
    def num(self) -> int:
        return int(self)

    @property
    def bin(self) -> bin:
        return bin(self)

    def binstr(self, width: Optional[int] = None) -> str:
        """ Binary representation of the state """
        return binstr(self, width or self.width)

    def binarr(self, width: Optional[int] = None,
               dtype: Optional[Union[int, str]] = None) -> np.ndarray:
        """ Returns the bits of an integer as a binary array. """
        return binarr(self, width or self.width, dtype)

    def binidx(self, width: Optional[int] = None) -> Iterable[int]:
        """ Indices of bits with value `1`. """
        return binidx(self, width or self.width)

    def count(self, value: int = 1) -> int:
        return bin(self).count(str(value))

    def get(self, bit: int) -> int:
        return int(self & (1 << bit))

    def flip(self, bit: int) -> 'Binary':
        return self.__class__(self ^ (1 << bit), width=self.width)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.binstr()})"

    def __str__(self) -> str:
        return self.binstr()

    # -------------- Math operators -------------

    def __add__(self, other: Union[int, 'Binary']) -> 'Binary':
        return self.__class__(super().__add__(other), width=self.width)

    def __radd__(self, other: Union[int, 'Binary']) -> 'Binary':
        return self.__class__(super().__radd__(other), width=self.width)

    def __sub__(self, other: Union[int, 'Binary']) -> 'Binary':
        return self.__class__(super().__sub__(other), width=self.width)

    def __rsub__(self, other: Union[int, 'Binary']) -> 'Binary':
        return self.__class__(super().__rsub__(other), width=self.width)

    def __mul__(self, other: Union[int, 'Binary']) -> 'Binary':
        return self.__class__(super().__mul__(other), width=self.width)

    def __rmul__(self, other: Union[int, 'Binary']) -> 'Binary':
        return self.__class__(super().__rmul__(other), width=self.width)

    # -------------- Binary operators -------------

    def __invert__(self) -> 'Binary':
        return self.__class__(super().__invert__(), width=self.width)

    def __and__(self, other: Union[int, 'Binary']) -> 'Binary':
        return self.__class__(super().__and__(other), width=self.width)

    def __or__(self, other: Union[int, 'Binary']) -> 'Binary':
        return self.__class__(super().__or__(other), width=self.width)

    def __xor__(self, other: Union[int, 'Binary']) -> 'Binary':
        return self.__class__(super().__xor__(other), width=self.width)

    def __lshift__(self, other: Union[int, 'Binary']) -> 'Binary':
        return self.__class__(super().__lshift__(other), width=self.width)

    def __rshift__(self, other: Union[int, 'Binary']) -> 'Binary':
        return self.__class__(super().__rshift__(other), width=self.width)


# =========================================================================
# State objects
# =========================================================================


class SpinState(Binary):

    def binstr(self, width: Optional[int] = None) -> str:
        """Returns the binary representation of the state"""
        return binstr(self, width)

    def binarr(self, width: Optional[int] = None,
               dtype: Optional[Union[int, str]] = None) -> np.ndarray:
        """Returns the bits of the integer as a binary array."""
        return binarr(self, width, dtype)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.binstr()})"

    def __str__(self) -> str:
        return self.binstr()

    @property
    def n(self) -> int:
        """Total occupation of the state"""
        return bin(self).count("1")

    def occ(self, pos: int) -> int:
        """Returns the occupation at index `pos`."""
        return self & (1 << pos)

    def occupations(self, dtype: Optional[Union[int, str]] = None) -> np.ndarray:
        """Returns the site occupations of a state as a binary array."""
        return occupations(self, dtype=dtype)

    def overlap(self, other: Union[int, 'SpinState'],
                dtype: Optional[Union[int, str]] = None) -> np.ndarray:
        """Computes the overlap with another state and returns the results as a binary array."""
        return overlap(self, other, dtype=dtype)

    def create(self, pos: int) -> 'SpinState':
        """Creates a particle at `pos` if possible and returns the new state."""
        return self.__class__(create(self, pos), width=self.width)

    def annihilate(self, pos: int) -> 'SpinState':
        """Annihilates a particle at `pos` if possible and returns the new state."""
        return self.__class__(annihilate(self, pos), width=self.width)


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

    def __iter__(self):
        return iter(self.states)


class Basis:
    """Container class all basis states the full Hilbert space of a model."""

    __slots__ = ["size", "num_sites", "num_spinstates", "sectors"]

    def __init__(self, num_sites: Optional[int] = 0, init_sectors: Optional[bool] = False):
        self.size = 0
        self.num_sites = 0
        self.num_spinstates = 0
        self.sectors = defaultdict(list)
        self.init(num_sites, init_sectors)

    @property
    def fillings(self):
        return list(range(self.num_sites))

    def init(self, num_sites: int, init_sectors: Optional[bool] = False):
        self.num_sites = num_sites
        self.num_spinstates = 2 ** num_sites
        self.size = self.num_spinstates ** 2
        self.sectors = defaultdict(list)
        if init_sectors:
            for state in range(2 ** num_sites):
                n = f"{state:b}".count("1")
                self.sectors[n].append(state)

    def generate_states(self, n: int = None):
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

    def get_states(self, n=None):
        if n in self.sectors:
            # Get cached spin-sector states
            states = self.sectors.get(n, list(range(self.num_spinstates)))
        else:
            # Compute new spin-sector states and store them
            states = self.generate_states(n)
            self.sectors[n] = states
        return states

    def get_sector(self, n_up=None, n_dn=None):
        up_states = self.get_states(n_up)
        dn_states = self.get_states(n_dn)
        return Sector(up_states, dn_states, n_up, n_dn, self.num_sites)

    def iter_fillings(self):
        return product(self.fillings, repeat=2)

    def iter_sectors(self):
        for n_up, n_dn in product(self.fillings, repeat=2):
            yield self.get_sector(n_up, n_dn)

    def __getitem__(self, item):
        if hasattr(item, "__len__"):
            return self.get_sector(*item)
        return self.get_states(item)

    def __repr__(self):
        return f"{self.__class__.__name__}(size: {self.size}, num_sites: {self.num_sites}, " \
               f"fillings: {self.fillings})"
