# coding: utf-8
#
# This code is part of cmpy.
# 
# Copyright (c) 2021, Dylan Jones

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import linalg as sla
from cmpy import operators, basis
from cmpy.models import SingleImpurityAndersonModel
from numpy.typing import ArrayLike


def main():
    siam = SingleImpurityAndersonModel(mu=None)
    hamop = siam.hamilton_operator()
    print(hamop.trace())

    sec = siam.get_sector(1, 1)
    data, indices = siam.hamiltonian_data(sec.up_states, sec.dn_states)
    indices = np.asarray(indices).T
    print(indices[:, 0])
    print(indices[:, 1])
    print(np.where(indices[:, 0] == indices[:, 1])[0])


if __name__ == "__main__":
    main()
