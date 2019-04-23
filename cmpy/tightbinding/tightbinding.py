# -*- coding: utf-8 -*-
"""
Created on 26 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
import numpy as np
from cmpy.core import Lattice, Hamiltonian
from cmpy.core import distance, chain, vlinspace
from cmpy.core import Progress


def shuffle(obj, disorder):
    """ Shuffle elements of object

    Parameters
    ----------
    obj: array_like or int
        object to shuffle
    disorder: float
        disorder amount

    Returns
    -------
    shuffled: array_like
    """
    if len(obj.shape) == 0:
        return obj + np.random.uniform(-disorder, +disorder)
    else:
        shuffled = obj.copy()
        delta = np.random.uniform(-disorder, +disorder)
        for i in range(shuffled.shape[0]):
            shuffled[i, i] += delta
        return shuffled


class TightBinding:

    def __init__(self, vectors=np.eye(2), lattice=None):
        self.lattice = Lattice(vectors) if lattice is None else lattice
        self.n_orbs = None
        self.energies = list()
        self.hoppings = list()

        # Cached hamiltonian objects for more performance
        self._cached_ham = None
        self._cached_slice = None
        self._cached_slice_energies = None
        self._cached_hop = None

    @property
    def n_base(self):
        """int: number of sites in unitcell of lattice """
        return self.lattice.n_base

    @property
    def n_elements(self):
        """int: number of energy elements (orbitals) in system"""
        return self.lattice.n * self.n_orbs

    @property
    def slice_elements(self):
        """int: number of energy elements (orbitals) in a slice of the system"""
        return self.lattice.slice_sites * self.n_orbs

    @property
    def channels(self):
        """int: number of transmission channels of the system"""
        return np.prod(self.lattice.shape)

    def copy(self):
        """TightBinding: Make a deep copy of the model"""
        latt = self.lattice.copy()
        tb = TightBinding(lattice=latt)
        tb.n_orbs = self.n_orbs
        tb.energies = self.energies
        tb.hoppings = self.hoppings
        return tb

    def add_atom(self, name="A", pos=None, energy=0.):
        """ Add site to lattice cell and store it's energy.

        Parameters
        ----------
        name: str, optional
            Name of the site. The defualt is the origin of the cell
        pos: array_like, optional
            position of site in the lattice cell, default: "A"
        energy: scalar or array_like, optional
            on-site energy of the atom, default: 0.

        Raises
        ------
        ValueError:
            raised if position is allready occupied or
            number of orbitals doesn*t match others.
        """
        self.lattice.add_atom(name, pos)
        energy = np.array(energy)
        n_orbs = energy.shape[0] if len(energy.shape) else 1
        if self.n_orbs is None:
            self.n_orbs = n_orbs
        elif n_orbs != self.n_orbs:
            raise ValueError(f"Orbit number doesn't match ({self.n_orbs})!")
        self.energies.append(energy)

    def set_hopping(self, *hoppings):
        """ Set hopping energies of the model

        Parameters
        ----------
        hoppings: list
            hopping energies for the first n neighbors. Elements can be
            either scalar or array, butr must match orbit number of on-site
            energies
        """
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
        """ Get the saved hopping value for the given distance

        Parameters
        ----------
        dist: float
            distance-value

        Returns
        -------
        t: np.ndarray
        """
        idx = self.lattice.distances.index(dist)
        return self.hoppings[idx]

    def get_energy(self, i):
        """ Get on-site energy of i-th site

        Parameters
        ----------
        i: int
            index of site

        Returns
        -------
        eps: np.ndarray
        """
        _, alpha = self.lattice.get(i)
        return self.energies[alpha]

    def clear_slice_cache(self):
        self._cached_slice = None
        self._cached_slice_energies = None
        self._cached_hop = None

    def clear_ham_cache(self):
        self._cached_ham = None

    def clear_cache(self):
        """Clear cached hamiltonian objects"""
        self.clear_ham_cache()
        self.clear_slice_cache()

    def build(self, shape=None):
        """ Build model

        Parameters
        ----------
        shape: tuple
            shape of the lattice
        """
        self.clear_cache()
        self.lattice.build(shape)

    def reshape(self, x=None, y=None, z=None):
        """ Reshape existing lattice build

        Parameters
        ----------
        x: int, optional
            new size in x-direction
        y: int, optional
            new size in y-direction
        z: int, optional
            new size in z-direction
        """
        self._cached_ham = None
        self.lattice.reshape(x, y, z)
        if (y is not None) or (z is not None):
            self.clear_slice_cache()

    def dispersion(self, k):
        """ Calculate dispersion of the model in k-space

        Parameters
        ----------
        k: float
            wave-vector

        Returns
        -------
        disp: np.ndarray
        """
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
        """ Calculate Bandstructure in k-space between given k-points

        Parameters
        ----------
        points: list
            k-points for the bandstructure
        n_e: int, optional
            number of energy samples in each section
        thresh: float, optional
            threshold to find energy-values, default: 1e-9
        scale: bool, optional
            scale lengths of sections, default: True
        verbose: bool, optional
            print progress if True, optional: True

        Returns
        -------
        band_sections: list
        """
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

    # =========================================================================
    # Hamiltonians
    # =========================================================================

    def _init_slice_hamiltonian(self):
        """ Calculate the hamiltonian of a slice of the model

        Returns
        -------
        ham: Hamiltonian
        """
        n = self.lattice.slice_sites
        ham = Hamiltonian.zeros(n, self.n_orbs, "complex")

        slice_energies = list()

        for i in range(n):
            _, alpha = self.lattice.get(i)
            # Site energies
            eps = self.energies[alpha]
            slice_energies.append(eps)
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

        self._cached_slice_energies = slice_energies
        self._cached_slice = ham

    def slice_hamiltonian(self, w_eps=0.):
        """ Get the slice hamiltonian of the model and add disorder

        Notes
        -----
        Uses cached data if available, otherwise cache will be initialized

        Parameters
        ----------
        w_eps: float, optional
            on-site disorder strength

        Returns
        -------
        ham: Hamiltonian
        """
        if self._cached_slice is None:
            self._init_slice_hamiltonian()
        ham = self._cached_slice

        n = self.lattice.slice_sites
        for i in range(n):
            eps = self._cached_slice_energies[i]
            if w_eps != 0:
                eps = shuffle(eps, w_eps)
            ham.set_energy(i, eps)

        return ham

    def _init_hop_hamiltonian(self):
        """ Calculate the hopping hamiltonian between two slices

        Returns
        -------
        ham: Hamiltonian
        """
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

    def slice_hopping(self):
        """ Get the hopping hamiltonian between two slices of the model

        Notes
        -----
        Uses cached data if available, otherwise cache will be initialized

        Parameters
        ----------
        w_eps: float, optional
            on-site disorder strength

        Returns
        -------
        ham: Hamiltonian
        """
        if self._cached_hop is None:
            self._cached_hop = self._init_hop_hamiltonian()
        return self._cached_hop

    def _init_hamiltonian(self, blocksize=None):
        """ Calculate the full hamiltonian of the model

        Parameters
        ----------
        blocksize: int, optional
            blocksize that the hamiltonian will be configured

        Returns
        -------
        ham: Hamiltonian
        """
        n = self.lattice.n
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
                for j in neighbours[dist]:
                    if j > i:
                        ham.set_hopping(i, j, t)
        if blocksize is not None:
            ham.config_blocks(blocksize)
        self._cached_ham = ham

    def hamiltonian(self, w_eps=0., blocksize=None):
        """ Get the full hamiltonian of the model and add disorder

        Notes
        -----
        Uses cached data if available, otherwise cache will be initialized

        Parameters
        ----------
        w_eps: float, optional
            on-site disorder strength

        Returns
        -------
        ham: Hamiltonian
        """
        if self._cached_ham is None:
            self._init_hamiltonian(blocksize)
        else:
            # Reset interface elements
            h_interface = self.slice_hamiltonian()
            self._cached_ham.config_blocks(self.slice_elements)
            n = self._cached_ham.block_shape[0] - 1
            self._cached_ham.set_block(0, 0, h_interface)
            self._cached_ham.set_block(n, n, h_interface)
        ham = self._cached_ham

        # Add disorder to hamiltonian

        # if w_eps:
        #     delta = np.eye(ham.shape[0]) * np.random.uniform(-w_eps, w_eps)
        #     ham = ham + delta

        n = self.lattice.n
        if w_eps:
            for i in range(n):
                _, alpha = self.lattice.get(i)
                # Site energies
                eps = shuffle(self.energies[alpha], w_eps)
                ham.set_energy(i, eps)
        return ham
