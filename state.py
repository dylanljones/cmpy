# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2018
@author: Dylan Jones

project: tmp$
version: 1.0
"""
import numpy as np
import scipy.linalg as la
from itertools import product
from sciutils import Plot, use_cycler
from cmpy import Hamiltonian
from cmpy.models import HubbardModel, TightBinding
from scipy.sparse import csr_matrix


def get_states(particles):
    n = 2 ** particles
    return [ManyBodyState(x) for x in range(n)]



class ManyBodyState(int):

    def __str__(self):
        return self.bin[::-1]

    def check(self, i):
        return (self >> i) & 1

    @property
    def bin(self):
        return str(bin(self)).replace("0b", "")

    @property
    def particles(self):
        return self.bin.count('1')

    def create(self, i):
        if self.check(i):
            return None
        return self.flip(i)

    def annihilate(self, i):
        if not self.check(i):
            return None
        return self.flip(i)

    def flip(self, i):
        return ManyBodyState(self ^ (1 << i))

    def array(self, n, spin=2):
        length = n * spin
        x = np.zeros(length)
        for i in range(length):
            if self.check(i):
                x[i] = 1
        return x


def destruct(particles, index):
    """ Creates a fermionic annihilation operator in matrix representation

    The phase +-1 is needed to ensure the operators follow the commutator-rules

    Parameters
    ----------
    particles: int
        Number of particles in the system
    index: int
        index of operator

    Returns
    -------
    mat: csr_matrix
        sparse matrix representation of operator
    """
    n = 2**particles
    mat = np.zeros((n, n))
    flipper = 2**index
    for state in range(n):
        # check if there is a spin at index i
        ispin = (state >> index) & 1
        if ispin == 1:
            amount = bin(state >> index+1).count('1')
            mat[state ^ flipper, state] = 1 if amount % 2 == 0 else -1
    return mat


def main():
    states = get_states(2)
    print(states[2])
    print(destruct(2, 0))


if __name__ == "__main__":
    main()
