# -*- coding: utf-8 -*-
"""
Created on 24 Apr 2019
@author: Dylan Jones

project: tightbinding
version: 1.0
"""
import os
import numpy as np
from cmpy import Matrix

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")
DFT_DATA = os.path.join(DATA_DIR, "dft")


def read_dft_file(file, num_headers=4):
    # Read raw data
    with open(file, "r") as f:
        lines = f.readlines()
    orbs1, orbs2 = list(), list()
    cell_pos, pos = list(), list()
    energies = list()
    for line in lines[num_headers:]:
        parts = line.split()
        orbs1.append(" ".join(parts[:2]))   # first atom and orbit
        orbs2.append(" ".join(parts[2:4]))  # second atom and orbit
        cell_pos.append(parts[4:7])         # cell position
        pos.append(parts[7:10])             # atom shift
        energies.append(parts[10])          # hopping parameter
    return orbs1, orbs2, cell_pos, pos, energies


def build_dft_data(file):
    raw_orbs1, raw_orbs2, raw_cell_pos, raw_atom_pos, energies = read_dft_file(file)
    orbs1, orbs1_idx = np.unique(raw_orbs1, return_inverse=True)
    orbs2, orbs2_idx = np.unique(raw_orbs2), list()
    for orb in raw_orbs2:
        i = np.where(orb == orbs1)[0][0]
        orbs2_idx.append(i)

    cell_pos, cell_pos_idx = np.unique(raw_cell_pos, axis=0, return_inverse=True)
    atom_pos, atom_pos_idx = np.unique(raw_atom_pos, axis=0, return_inverse=True)
    n, n_cells = orbs1.shape[0], cell_pos.shape[0]

    energie_mat = np.zeros((n, n, n_cells))
    cell_pos_mat = np.zeros((n, n, n_cells, 3))
    atom_pos_mat = np.zeros((n, n, n_cells, 3))

    for idx, t in enumerate(energies):
        cpos = raw_cell_pos[idx]
        apos = raw_atom_pos[idx]

        i, j = orbs1_idx[idx], orbs2_idx[idx]
        i_cell = cell_pos_idx[idx]

        energie_mat[i, j, i_cell] = t
        cell_pos_mat[i, j, i_cell, :] = cpos
        atom_pos_mat[i, j, i_cell, :] = apos

    return energie_mat, cell_pos_mat, atom_pos_mat, orbs1


def dft_ham(file):
    energie_mat, cell_pos_mat, atom_pos_mat, orbs = build_dft_data(file)
    m = Matrix(np.sum(energie_mat, axis=2))
    plot = m.show(show=False)
    plot.set_ticklabels(orbs)
    plot.tight()
    plot.show()



def main():
    file = os.path.join(DFT_DATA, "HAR1")
    dft_ham(file)



if __name__ == "__main__":
    main()
