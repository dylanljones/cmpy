# -*- coding: utf-8 -*-
"""
Created on 16 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np


class Binary:

    def __init__(self, x=0):
        """ Initialize integer-bits

        Parameters
        ----------
        x: int or Binary or str, optional
            Integer value. Default is 0.
        """
        if isinstance(x, Binary):
            x = x.int
        elif isinstance(x, str):
            x = int(x, 2)
        self.int = int(x)

    @classmethod
    def from_array(cls, array):
        self = cls()
        for i, val in enumerate(array):
            self.set_bit(i, val)
        return self

    @property
    def bin(self):
        return str(bin(self.int)).replace("0b", "")

    @property
    def size(self):
        return len(self.bin)

    def __len__(self):
        return len(self.bin)

    def __repr__(self):
        return self.bin

    @staticmethod
    def _get_value(other):
        if isinstance(other, Binary):
            return other.int
        return other

    @classmethod
    def _new(cls, x):
        return cls(x)

    def get_bit(self, idx):
        return ((self >> idx) & 1).int

    def set_bit(self, idx, value):
        mask = 1 << idx
        self.int &= ~mask
        if value:
            self.int |= mask

    def shift_add(self, val):
        self.int = int((self << 1) + val)

    def __getitem__(self, item):
        return self.get_bit(item)

    def __setitem__(self, key, value):
        self.set_bit(key, value)

    def __int__(self):
        return self.int

    # -------------- Comparisons -------------

    def __eq__(self, other):
        return self.int == self._get_value(other)

    def __ne__(self, other):
        return self.int != self._get_value(other)

    def __neg__(self):
        return -self.int

    def __lt__(self, other):
        return self.int < self._get_value(other)

    def __le__(self, other):
        return self.int <= self._get_value(other)

    def __gt__(self, other):
        return self.int > self._get_value(other)

    def __ge__(self, other):
        return self.int >= self._get_value(other)

    # -------------- Math operators -------------

    def __add__(self, other):
        return self._new(self.int + self._get_value(other))

    def __radd__(self, other):
        return self._new(self.int + self._get_value(other))

    def __sub__(self, other):
        return self._new(self.int - self._get_value(other))

    def __rsub__(self, other):
        return self._new(self._get_value(other) - self.int)

    def __mul__(self, other):
        return self._new(self.int * self._get_value(other))

    def __rmul__(self, other):
        return self._new(self.int * self._get_value(other))

    def __truediv__(self, other):
        return self._new(self.int / self._get_value(other))

    def __rtruediv__(self, other):
        return self._new(self._get_value(other) / self.int)

    # -------------- Binary operators -------------

    def __invert__(self):
        return self._new(~self.int)

    def __and__(self, other):
        return self._new(self.int & self._get_value(other))

    def __or__(self, other):
        return self._new(self.int | self._get_value(other))

    def __xor__(self, other):
        return self._new(self.int ^ self._get_value(other))

    def __lshift__(self, other):
        return self._new(self.int << self._get_value(other))

    def __rshift__(self, other):
        return self._new(self.int >> self._get_value(other))

    # ===================================================

    def count(self, bit=1):
        return self.bin.count(str(bit))

    def flip(self, i):
        return self._new(self.int ^ (1 << i))

    def indices(self):
        return [i for i in range(self.size) if self.get_bit(i) == 1]

    def array(self, size=None):
        size = size or self.size
        array = np.zeros(size, dtype="int")
        for i in range(self.size):
            if self.get_bit(i):
                array[i] = 1
        return array
