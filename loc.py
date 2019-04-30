# -*- coding: utf-8 -*-
"""
Created on 3 Mar 2018
@author: Dylan Jones

project: tightbinding
version: 1.0
"""
import os
import numpy as np
from cmpy import eta, Folder, DATA_DIR
from cmpy.tightbinding import TbDevice, s_basis, p3_basis, sp3_basis
from cmpy.tightbinding import disorder_lt, calculate_lt, loc_length
from cmpy.tightbinding.loclength import LT_Data, calculate_lt

ROOT = os.path.join(DATA_DIR, "localization")


def calculate_disorder_lt(basis, w_values, h, lengths=None, e=0, n_avrg=250):
    # initialize folder
    rel_parts = [ROOT]
    if basis.n == 1:
        filename = f"disorder-e={e}-h={h}.npz"
        rel_parts.append("s-basis")
    else:
        soc = basis.soc
        filename = f"disorder-e={e}-h={h}-soc={soc}.npz"
        if basis.n == 6:
            rel_parts.append("p3-basis")
        elif basis.n == 8:
            rel_parts.append("sp3-basis")
        rel_parts.append(f"soc={soc}")
    folder = Folder(*rel_parts)
    path = folder.build_path(filename)
    return disorder_lt(path, basis, h, w_values, lengths=lengths, e=e, n_avrg=n_avrg)


def calculate_batched(basis, w_values, heights, n_batch=500, n_avrg=3000):
    for n in range(n_batch, n_avrg + 1, n_batch):
        for h in heights:
            calculate_disorder_lt(basis, w_values, h, n_avrg=n)


def calculate_s_basis(n_avrg=1000):
    h = 16
    basis = s_basis()
    w_values = np.arange(16) + 1
    calculate_disorder_lt(basis, w_values, h)


def calculate(n_avrg=100):
    soc_values = 1, 2, 3, 4, 5, 7, 10
    heights = [1, 4, 8, 16]
    w_values = np.arange(16) + 1
    for soc in soc_values:
        for h in heights:
            basis = p3_basis(eps_p=0, t_pps=1, t_ppp=1, soc=soc)
            # calculate
            calculate_disorder_lt(basis, w_values, h, n_avrg=n_avrg)



def main():
    calculate()
    return
    soc = 1

    basis = s_basis()
    #basis = p3_basis(eps_p=0, t_pps=1, t_ppp=1, soc=soc)
    # basis = sp3_basis(soc=0)

    w_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    heights = [1, 4, 8] # [1, 2, 4, 6]
    # update_all_lengths(S_PATH, 10)

    calculate_batched(basis, w_values, heights, n_avrg=2000, n_batch=1000)


if __name__ == "__main__":
    main()
