# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import numpy as np
from .abc import AbstractSpinModel


class HeisenbergModel(AbstractSpinModel):

    def __init__(self, latt, j=1.):
        super().__init__(latt.num_sites)
        self.latt = latt
        self.j = j

    def _hamiltonian_data(self, states):
        neighbors = [self.latt.neighbors(pos1) for pos1 in range(self.num_sites)]
        for idx1, s1 in enumerate(states):
            for pos1 in range(self.num_sites):
                for pos2 in neighbors[pos1]:
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

    def hamiltonian(self, s=None, states=None, dtype=None):
        if states is None:
            states = self.get_states(s)
        shape = len(states), len(states)
        ham = np.zeros(shape, dtype)
        for i, j, val in self._hamiltonian_data(states):
            ham[i, j] += val
        return ham