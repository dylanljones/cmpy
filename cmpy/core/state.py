# -*- coding: utf-8 -*-
"""
Created on 14 Jul 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from sciutils import Binary
from .operators import annihilators

UP_CHAR = "\u2191"
DOWN_CHAR = "\u2193"
DOUBLE_CHAR = "d"
EMPTY_CHAR = "."
CHARS = {(1, 1): DOUBLE_CHAR, (0, 0): EMPTY_CHAR,
         (1, 0): UP_CHAR, (0, 1): DOWN_CHAR}

# =========================================================================
#                             DUAL STATE
# =========================================================================


class State2:

    def __init__(self, up=0, down=0, size=None):
        """

        Parameters
        ----------
        up: int or Binary or str, optional
        down: int or Binary or str, optional
        size: int, optional
        """
        self.up, self.down = Binary(up), Binary(down)
        self.size = size or max(self.up.size, self.down.size)

    @classmethod
    def from_bin(cls, val):
        state = Binary(val)
        u, d = Binary(), Binary()
        for i in range(0, state.size, 2):
            u.shift_add(state.get_bit(i))
            d.shift_add(state.get_bit(i+1))
        return cls(u, d)

    @classmethod
    def single_up(cls, i, size=None):
        u = 1 << i
        return cls(u, 0, size)

    @classmethod
    def single_down(cls, i, size=None):
        d = 1 << i
        return cls(0, d, size)

    @property
    def particles(self):
        return self.up.count(1) + self.down.count(1)

    @property
    def spin(self):
        return self.up.count(1) - self.down.count(1)

    @property
    def bin(self):
        state = Binary()
        for i in range(self.size):
            state.shift_add((self.up >> i) & 1)
            state.shift_add((self.down >> i) & 1)
        return state

    def __repr__(self):
        u, d = str(self.up), str(self.down)
        return f"State({u:0>{self.size}}, {d:0>{self.size}})"

    def __eq__(self, other):
        return self.up == other.up and self.down == other.down

    def __lt__(self, other):
        return self.bin < other.bin

    def __le__(self, other):
        return self.bin <= other.bin

    def __gt__(self, other):
        return self.bin > other.bin

    def __ge__(self, other):
        return self.bin >= other.bin

    def site_char(self, i):
        occ = (self.up.get_bit(i), self.down.get_bit(i))
        return CHARS[occ]

    def label(self, n=None):
        n = n or self.size
        string = ""
        for i in range(n):
            string += self.site_char(i)
        return string

    def copy(self):
        return State2(self.up, self.down)

    # =========================================================================

    def create(self, i, spin=0):
        u, d = self.up, self.down
        if spin == 0:
            if u.get_bit(i) == 1:
                return None
            u = u.flip(i)
        else:
            if d.get_bit(i) == 1:
                return None
            d = d.flip(i)
        return State2(u, d)

    def annihilate(self, i, spin=0):
        u, d = self.up, self.down
        if spin == 0:
            if u.get_bit(i) == 0:
                return None
            u = u.flip(i)
        else:
            if d.get_bit(i) == 0:
                return None
            d = d.flip(i)
        return State2(u, d)

    def spinarrays(self):
        return np.asarray([self.up.array(self.size), self.down.array(self.size)])

    def occupations(self):
        return self.spinarrays().sum(axis=0)

    def difference(self, other):
        up = self.up ^ other.up
        down = self.down ^ other.down
        return State2(up, down, self.size)

    def check_hopping(self, other):
        up_diff = (self.up ^ other.up).indices()
        down_diff = (self.down ^ other.down).indices()
        if len(up_diff) == 2 and len(down_diff) == 0:
            return up_diff
        elif len(up_diff) == 0 and len(down_diff) == 2:
            return down_diff
        return None

    def phase(self, i):
        state = self.bin
        particles = (state >> i + 1).count(1)
        return 1 if particles % 2 == 0 else -1


# =========================================================================
#                               STATE
# =========================================================================

class State(Binary):

    @classmethod
    def from_spinarrays(cls, up, down):
        n = len(up)
        arr = np.array([up, down]).T.reshape(2 * n)
        return cls.from_array(arr)

    @classmethod
    def single(cls, i):
        return cls(1) << i

    @property
    def particles(self):
        return self.bin.count('1')

    @property
    def spin(self):
        arr = self.array()
        up = arr[0:arr.size:2]
        down = arr[1:arr.size:2]
        return np.sum(up) - np.sum(down)

    def copy(self):
        return State(self)

    def spinarrays(self, n):
        arr = self.array(2 * n)
        return arr.reshape(n, 2).T

    @staticmethod
    def bitindex(idx, spin):
        return 2 * idx + spin

    def site_char(self, i):
        occ = self.get_bit(2*i), self.get_bit(2*i + 1)
        return CHARS[occ]

    def label(self, n):
        string = ""
        for i in range(n):
            string += self.site_char(i)
        return string

    # =========================================================================

    def create(self, i):
        if self.get_bit(i) == 1:
            return None
        return self.flip(i)

    def annihilate(self, i):
        if self.get_bit(i) == 0:
            return None
        return self.flip(i)

    def occupied(self, i):
        return bool(self.get_bit(i))

    def occupations(self, n):
        return self.spinarrays(n).sum(axis=0)

    def difference(self, other):
        return self ^ other

    def check_hopping(self, other):
        if self.particles != other.particles:
            return None
        if self.spin != other.spin:
            return None
        idx = self.difference(other).indices()
        if len(idx) != 2:
            return None
        i, j = idx
        return i, j

    def phase(self, i):
        particles = (self >> i + 1).particles
        return 1 if particles % 2 == 0 else -1


def free_basis(n):
    states = [State(0)]
    for i in range(n):
        states.append(State(1 << i))
    return states


def fock_basis(particles):
    n = 2 ** (2 * particles)
    return [State(x) for x in range(n)]


def state_indices(states, particles=None):
    n = len(states)
    indices = list(range(n))
    if particles is not None:
        for i in range(n):
            if states[i].particles not in particles:
                indices.remove(i)
    return np.asarray(indices)


class Basis:

    def __init__(self, n=2, states=None):
        self.n = n
        self.states = states or fock_basis(self.n)
        self.ops = None
        self._build_operators()

    def _build_operators(self):
        self.ops = annihilators(self.states, 2 * self.n)

    @property
    def c_up(self):
        return self.ops[::2]

    @property
    def c_down(self):
        return self.ops[1::2]

    @property
    def labels(self):
        return [s.label(self.n) for s in self.states]

    def sort_states(self, *args, **kwargs):
        self.states = sorted(self.states, *args, **kwargs)
        self._build_operators()

    def indices(self, particles=None):
        n = len(self.states)
        indices = list(range(n))
        if particles is not None:
            for i in range(n):
                if self.states[i].particles not in particles:
                    indices.remove(i)
        return np.asarray(indices)

    def subbasis(self, particles=None):
        states = [self.states[i] for i in self.indices(particles)]
        return Basis(self.n, states)
