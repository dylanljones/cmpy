# -*- coding: utf-8 -*-
"""
Created on 26 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
import numpy as np
from cmpy.core import Lattice, Hamiltonian, SparseHamiltonian
from cmpy.core import distance, shuffle, chain, vlinspace
from cmpy.core import Progress


class TightBinding:

    def __init__(self, vectors=np.eye(2)):
        self.lattice = Lattice(vectors)
        self.n_orbs = None
        self.energies = list()
        self.hoppings = list()

        self.sparse_mode = False

    @property
    def n_base(self):
        return self.lattice.n_base

    @property
    def n_elements(self):
        return self.lattice.n * self.n_orbs

    @property
    def slice_elements(self):
        return self.lattice.slice_sites * self.n_orbs

    @property
    def channels(self):
        return np.prod(self.lattice.shape)

    def set_sparse_mode(self, enabled=False):
        self.sparse_mode = enabled

    def add_atom(self, name="A", pos=None, energy=0.):
        self.lattice.add_atom(name, pos)
        energy = np.array(energy)
        n_orbs = energy.shape[0] if len(energy.shape) else 1
        if self.n_orbs is None:
            self.n_orbs = n_orbs
        elif n_orbs != self.n_orbs:
            raise ValueError(f"Orbit number doesn't match ({self.n_orbs})!")
        self.energies.append(energy)

    def set_hopping(self, *hoppings):
        self.hoppings = list()
        for t in hoppings:
            if not hasattr(t, "__len__"):
                t = [[t]]
            t = np.asarray(t)
            n_orbs = t.shape[0] if len(t.shape) else 1
            if n_orbs != self.n_orbs:
                raise ValueError(f"Orbit number doesn't match ({self.n_orbs})!")
            self.hoppings.append(t)
        self.lattice.calculate_distances(len(self.hoppings))

    def get_hopping(self, dist):
        idx = self.lattice.distances.index(dist)
        return self.hoppings[idx]

    def get_energy(self, i):
        _, alpha = self.lattice.get(i)
        return self.energies[alpha]

    def build(self, shape=None):
        self.lattice.build(shape)

    def dispersion(self, k):
        ham = Hamiltonian.zeros(self.n_base, self.n_orbs, "complex")
        for i in range(self.n_base):
            r1 = self.lattice.atom_positions[i]
            nn_vecs = self.lattice.neighbour_vectors(alpha=i)
            eps_i = self.energies[i]
            t = self.hoppings[0]
            if self.n_base == 1:
                eps = eps_i + t * np.sum([np.exp(1j * np.dot(k, v)) for v in nn_vecs])
                ham.set_energy(i, eps)
            else:
                ham.set_energy(i, eps_i)

            for j in range(i+1, self.n_base):
                r2 = self.lattice.atom_positions[i][j]
                dist = distance(r1, r2)
                if dist in self.lattice.distances:
                    vecs = r2-r1, r1-r2
                    t = self.get_hopping(dist) * sum([np.exp(1j * np.dot(k, v)) for v in vecs])
                    ham.set_hopping(i, j, t)
        return ham.eigvals()

    def bands(self, points, n_e=1000, thresh=1e-9, scale=True, verbose=True):
        pairs = list(chain(points, cycle=True))
        n_sections = len(pairs)
        if scale:
            distances = [distance(p1, p2) for p1, p2 in pairs]
            sect_sizes = [int(n_e * dist / max(distances)) for dist in distances]
        else:
            sect_sizes = [n_e] * n_sections
        n = sum(sect_sizes)

        band_sections = list()
        with Progress(total=n, header="Calculating dispersion", enabled=verbose) as p:
            for i in range(n_sections):
                p1, p2 = pairs[i]
                n_points = sect_sizes[i]
                e_vals = np.zeros((n_points, self.n_base * self.n_orbs))
                k_vals = vlinspace(p1, p2, n_points)
                for j in range(n_points):
                    p.update(f"Section {i+1}/{n_sections}")
                    disp = self.dispersion(k_vals[j])
                    indices = np.where(np.isclose(disp, 0., atol=thresh))[0]
                    disp[indices] = np.nan
                    e_vals[j] = disp
                band_sections.append(e_vals)
        return band_sections

    def hamiltonian(self, w_eps=0.):
        n = self.lattice.n
        if self.sparse_mode:
            ham = SparseHamiltonian.zeros(n, self.n_orbs, "complex")
        else:
            ham = Hamiltonian.zeros(n, self.n_orbs, "complex")

        # Set values
        for i in range(n):
            n, alpha = self.lattice.get(i)

            # Site energies
            eps = self.energies[alpha]
            if w_eps:
                eps = shuffle(eps, w_eps)
            ham.set_energy(i, eps)

            # Hopping energies
            neighbours = self.lattice.neighbours[i]
            for dist in range(len(neighbours)):
                t = self.hoppings[dist]
                indices = neighbours[dist]
                for j in indices:
                    if j > i:
                        ham.set_hopping(i, j, t)
        return ham

    def slice_hamiltonian(self):
        n = self.lattice.slice_sites
        ham = Hamiltonian.zeros(n, self.n_orbs, "complex")
        for i in range(n):
            n, alpha = self.lattice.get(i)
            # Site energies
            eps = self.energies[alpha]
            ham.set_energy(i, eps)
            # Hopping energies
            neighbours = self.lattice.neighbours[i]
            for dist in range(len(neighbours)):
                t = self.hoppings[dist]
                indices = neighbours[dist]
                for j in indices:
                    try:
                        ham.set_hopping(i, j, t)
                    except ValueError:
                        pass
        return ham

    def slice_hopping(self):
        n = self.lattice.slice_sites
        ham = Hamiltonian.zeros(n, self.n_orbs, "complex")
        for i in range(n):
            # Hopping energies
            for dist in range(len(self.lattice.distances)):
                for j in range(n):
                    if j + n in self.lattice.neighbours[i][dist]:
                        t = self.hoppings[dist]
                        ham.set(i, j, t)
        return ham
