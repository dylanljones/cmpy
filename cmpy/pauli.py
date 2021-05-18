#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.sparse.linalg import LinearOperator
import numpy as np


class Pauli(LinearOperator):

    def __init__(self, operator, dtype=None):
        super().__init__(dtype, operator.shape)
        self.operator = operator

    def _matvec(self, x):
        return np.dot(self.operator, x)


def main():
    pauli_x = np.array([[0,1j],[-1j,0]])
    paul = Pauli(pauli_x)
    print(paul @ np.eye(2))
    print(paul * np.ones(2))
    print(paul.adjoint() @ np.eye(2))

if __name__ == "__main__":
    main()
