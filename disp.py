# -*- coding: utf-8 -*-
"""
Created on 26 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
import numpy as np
from sciutils import *
from cmpy import *


def dispersion(model, k):
    """ Calculate dispersion of the model in k-space

    Parameters
    ----------
    k: float
        wave-vector

    Returns
    -------
    disp: np.ndarray
    """
    ham = model.cell_hamiltonian()
    if model.n_base == 1:
        dist = 0
        nn_vecs = model.lattice.neighbour_vectors(alpha=0, dist_idx=dist)
        t = model.get_hopping(atom1=0, atom2=0, distidx=dist)
        ham = ham + t * np.sum([np.exp(1j * np.dot(k, v)) for v in nn_vecs])
    else:
        pass
    return ham.eigvals()

    # ham = TbHamiltonian.zeros(self.n_base, self.n_orbs, "complex")
    # for i in range(self.n_base):
    #     r1 = self.lattice.atom_positions[i]
    #     nn_vecs = self.lattice.neighbour_vectors(alpha=i)
    #     eps_i = self.energies[i]
    #     t = self.get_hopping(0)
    #     if self.n_base == 1:
    #         eps = eps_i + t * np.sum([np.exp(1j * np.dot(k, v)) for v in nn_vecs])
    #         ham.set_energy(i, eps)
    #     else:
    #         ham.set_energy(i, eps_i)
    #
    #     for j in range(i+1, self.n_base):
    #         r2 = self.lattice.atom_positions[i][j]
    #         dist = distance(r1, r2)
    #         if dist in self.lattice.distances:
    #             vecs = r2-r1, r1-r2
    #             t = self.get_hopping(dist) * sum([np.exp(1j * np.dot(k, v)) for v in vecs])
    #             ham.set_hopping(i, j, t)
    # return ham.eigvals()


def test_disp():
    orbs = "p_x", "p_y"
    eps = 0.5
    t = 1
    model = TightBinding()
    # model.add_atom(energy=[0.5, 0.75], orbitals=orbs)
    # model.add_atom(pos=[0.5, 0], energy=eps, orbitals=orbs)
    # model.set_hopping(t, "p")
    model.add_atom(energy=eps)
    model.add_atom(energy=-eps, pos=[0.5, 0])
    model.set_hopping(t)
    model.build((4, 1))

    gamma = np.zeros(2)
    x = np.array([np.pi, 0])

    points = -x, gamma, x
    names = "$-X$", "$\Gamma$", "$X$"
    model.plot_bands(points, names, cycle=False)


class MatrixBlocks:

    def __init__(self, shape, block_sizes):
        self._matrix_shape = shape
        self.slices = None

        self.set_blocks(block_sizes)

    @property
    def shape(self):
        return len(self.slices), len(self.slices[0])

    def set_blocks(self, block_sizes):
        self.slices = blockmatrix_slices(self._matrix_shape, block_sizes)

    def __getitem__(self, idx):
        return self.slices[idx[0]][idx[1]]




def main():
    test_disp()





if __name__ == "__main__":
    main()
