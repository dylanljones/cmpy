# -*- coding: utf-8 -*-
"""
Created on 26 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
import itertools
import numpy as np
from .utils import vrange, iter_indices, index_array
from .utils import translate, distance
from .printing import prange
from .plotting import LatticePlot

# np.random.seed(100)


class AppendError(Exception):

    def __init__(self, msg):
        super().__init__("Can't append lattice: " + msg)


class Lattice:

    def __init__(self, vectors):
        """ Initialize general bravais Lattice

        Parameters
        ----------
        vectors: array_like
            Primitive vectors of lattice.
        """
        self.vectors = np.asarray(vectors)
        self.dim = len(vectors)
        self.distances = list()
        self.atoms = list()
        self.atom_positions = list()

        # Chached data if lattice is built
        self.n = 0
        self.shape = np.zeros(self.dim)
        self.indices = None
        self.neighbours = list()

    @property
    def n_base(self):
        """ int: number of sites in unit-cell"""
        return len(self.atoms)

    def copy(self):
        """ Create new lattice with equivalent setup

        Returns
        -------
        latt: Lattice
        """
        latt = Lattice(self.vectors)
        for i in range(self.n_base):
            latt.add_atom(self.atoms[i], self.atom_positions[i])
        latt.calculate_distances(len(self.distances))
        if self.n:
            latt.build(self.shape)
        return latt

    def add_atom(self, name="A", pos=None):
        """ Add site to lattice cell.

        Parameters
        ----------
        name: str, optional
            Name of the site.
        pos: array_like, optional
            position of site in the lattice cell.

        Raises
        ------
        ValueError:
            raised if position is allready occupied or
            number of orbitals doesn*t match others.
        """
        if pos is None:
            pos = np.zeros(self.vectors.shape[0])
        if any(np.all(pos == x) for x in self.atom_positions):
            raise ValueError(f"Position {pos} allready occupied")
        self.atoms.append(name)
        self.atom_positions.append(pos)

    def get_atom(self, alpha):
        """ Returns atom name of unit cell.

        Parameters
        ----------
        alpha: int
            index of site.

        Returns
        -------
        atom: str
        """
        return self.atoms[alpha]

    def get_index(self, pos):
        """ Returns lattice index (n, alpha) for global position.

        Parameters
        ----------
        pos: array_like
            global site position.

        Returns
        -------
        idx: tuple
            lattice index consisting of translation vector n
            and site index alpha.
        """
        pos = np.asarray(pos)
        n = np.asarray(np.floor(pos @ np.linalg.inv(self.vectors)), dtype="int")
        r_alpha = pos - (self.vectors @ n)
        alpha = np.where((self.atom_positions == r_alpha).all(axis=1))[0]
        idx = n, alpha
        return idx

    def get_position(self, n, alpha=0):
        """ Returns position for a given translation vector and site index

        Parameters
        ----------
        n: np.ndarray
            translation vector.
        alpha: int, optional
            site index, default is 0.
        Returns
        -------
        pos: np.ndarray
        """
        r = self.atom_positions[alpha]
        return r + translate(self.vectors, n)

    def translate_cell(self, n):
        """ Translate all contents of unit cell

        Parameters
        ----------
        n: np.ndarray
            translation vector.

        Yields
        -------
        pos: np.ndarray
            positions of the sites in the translated unit cell
        """
        for alpha in range(self.n_base):
            yield self.get_position(n, alpha)

    def distance(self, idx0, idx1):
        """ Calculate distance between two sites

        Parameters
        ----------
        idx0: tuple
            lattice vector (n, alpha) of first site
        idx1: tuple
            lattice index (n, alpha) of second site

        Returns
        -------
        distance: float
        """
        r1 = self.get_position(*idx0)
        r2 = self.get_position(*idx1)
        return distance(r1, r2)

    def neighbour_range(self, n=None, cell_range=1):
        """ Get all neighbouring translation vectors of a given cell position

        Parameters
        ----------
        n: array_like, optional
            translation vector of unit cell, the default is the origin.
        cell_range: int, optional
            Range of neighbours, the default is 1.
        Returns
        -------
        trans_vectors: list
        """
        n = np.zeros(self.dim) if n is None else n
        offset = cell_range + 2
        ranges = [np.arange(n[d] - offset, n[d] + offset + 1) for d in range(self.dim)]
        n_vecs = vrange(ranges)
        return iter_indices(n_vecs, self.n_base)

    def get_neighbours(self, n=None, alpha=0, dist_idx=0, array=False):
        """ Find all neighbours of given site and return the lattice indices.

        Parameters
        ----------
        n: array_like, optional
            translation vector of site, the default is the origin.
        alpha: int, optional
            site index, default is 0.
        dist_idx: int, default
            index of distance to neighbours, defauzlt is 0 (nearest neighbours).
        array: bool, optional
            if true, return lattice index (n, alpha) as single array.
            The default is False.

        Returns
        -------
        indices: list
        """
        n = np.zeros(self.dim) if n is None else n
        idx = n, alpha
        dist = self.distances[dist_idx]
        indices = list()
        for idx1 in self.neighbour_range(n, dist_idx + 1):
            if np.isclose(self.distance(idx, idx1), dist, atol=1e-5):
                if array:
                    idx1 = [*idx1[0], idx1[1]]
                indices.append(idx1)
        return indices

    def neighbour_vectors(self, n=None, alpha=0, dist_idx=0):
        """ Find all neighbours of given site and return the vectors to them.

        Parameters
        ----------
        n: array_like, optional
            translation vector of site, the default is the origin.
        alpha: int, optional
            site index, default is 0.
        dist_idx: int, default
            index of distance to neighbours, defauzlt is 0 (nearest neighbours).

        Returns
        -------
        vectors: list
        """
        n = np.zeros(self.dim) if n is None else n
        r = self.get_position(n, alpha)
        dist = self.distances[dist_idx]
        vectors = list()
        for idx1 in self.neighbour_range(n, dist_idx + 1):
            r1 = self.get_position(*idx1)
            if distance(r, r1) == dist:
                vectors.append(r1-r)
        return vectors

    def calculate_distances(self, n=1):
        """ Calculate n lowest distances between sites in bravais lattice.

        Checks distances between all sites of the bravais lattice and saves n lowest values
        for later use. This speeds up many calculations like finding nearest neighbours.

        Parameters
        ----------
        n: int, optional
            number of distances of lattice structure to calculate.
            The default is 1 (nearest neighbours).
        """
        n += 1
        n_vecs = vrange(self.dim * [np.arange(-n, n)])
        r_vecs = [self.get_position(*idx) for idx in iter_indices(n_vecs, self.n_base)]
        pairs = list(itertools.product(r_vecs, self.atom_positions))
        distances = list({distance(r1, r2) for r1, r2 in pairs})
        distances.sort()
        self.distances = distances[1:n]

    # =========================================================================

    @property
    def slice_shape(self):
        """ np.ndarray: shape of slice of built lattice"""
        return self.shape[1:]

    @property
    def slice_sites(self):
        """ int: number of sites in slice of built lattice"""
        return np.prod(self.slice_shape) * self.n_base

    def get(self, i):
        """ Get lattice index of a site  with index i in the built lattice

        Parameters
        ----------
        i: int
            index of chached site

        Returns
        -------
        index: tuple
        """
        idx = self.indices[i]
        return idx[:-1], idx[-1]

    def position(self, i):
        """ Returns the position of a chached site with index i

        Parameters
        ----------
        i: int
            index of chached site

        Returns
        -------
        pos: np.ndarray
        """
        n, alpha = self.get(i)
        return self.get_position(n, alpha)

    def build_section(self, n_vecs):
        """ Calculate lattice indices (n, alpha) and indices of cached neighbours

        Translates the sites of the unit cell for all given translation vectors and
        calculate the lattice indices (n, alpha) of the site. After the sites are built,
        the indices of the cached neighbours are calculated for each cached site.

        Parameters
        ----------
        n_vecs: array_like
            translation vectors of the section

        Returns
        -------
        indices: list
            lattice indices (n, alpha) of all built sites
        neighbours: list
            index of neighbours in chaed site list
        """
        num_sites = len(n_vecs) * self.n_base

        # Build lattice indices from translation vectors
        indices = index_array(n_vecs, self.n_base)

        # Find neighbours of each site in the "indices" list
        neighbours = list()
        header = f"Analyzing sites ({num_sites})"
        for i in prange(num_sites, header=header, enabled=num_sites >= 5000):
            idx = indices[i]
            n, alpha = idx[:-1], idx[-1]
            site = i
            # get all cached sites (existing and new)
            if self.indices is not None:
                all_sites = np.append(self.indices, indices, axis=0)
                site += self.n
            else:
                all_sites = indices
            
            # Get relevant index range to only look for neighbours
            # in proximity of site (larger then highest distance)
            n_dist = len(self.distances)

            offset = int((n_dist + 1) * self.slice_sites)
            i0 = max(site - offset, 0)
            i1 = min(site + offset, self.n + len(indices))
            
            # Get neighbour indices of site in proximity
            neighbour_indices = list()
            for i_dist in range(n_dist):
                # Get neighbour indices of site for distance level
                dist_neighbours = list()
                for idx in self.get_neighbours(n, alpha, i_dist, array=True):
                    site_window = all_sites[i0:i1]
                    # Find site of neighbour and store if in cache
                    hop_idx = np.where(np.all(site_window == idx, axis=1))[0]
                    if hop_idx:
                        dist_neighbours.append(hop_idx[0] + i0)
                neighbour_indices.append(dist_neighbours)
                        
            # Add all cached neighbours to neighbourlist of site
            neighbours.append(neighbour_indices)
            
        return indices, neighbours

    def build(self, shape=None):
        """ Build cache of finite size lattice with given shape
        
        Parameters
        ----------
        shape: array_like
            shape of finite size lattice
        """
        if shape is None:
            shape = np.ones(self.dim, dtype="int")
        self.indices = None
        self.neighbours = None
        self.shape = np.asarray(shape)
        self.n = np.prod(shape) * self.n_base
        n_vecs = vrange([range(s) for s in shape])
        self.indices, self.neighbours = self.build_section(n_vecs)

    def reshape(self, x=None, y=None, z=None):
        """ Reshape the cached lattice structure

        Parameters
        ----------
        x: int, optional
            new size in x-direction
        y: int, optional
            new size in y-direction
        z: int, optional
            new size in z-direction
        """
        # Only length changed
        if (x is not None) and (y is None) and (z is None):
            delta_x = x - self.shape[0]
            # Adding more slices
            if delta_x > 0:
                self.add_slices(delta_x)
                return

        # If none of the above cases hold up, build full lattice
        shape = self.shape
        for i, val in enumerate([x, y, z]):
            if val is not None:
                shape[i] = val
        self.build(shape)

    def add_slices(self, n):
        """ Add n slices of allready built lattice cache
        
        Parameters
        ----------
        n: int
            number of slices to add
        """
        new_xrange = self.shape[0] + np.arange(n)
        ranges = [new_xrange] + [range(s) for s in self.slice_shape]
        new_vecs = vrange(ranges)
        idx, neighbour_idx, = self.build_section(new_vecs)
        self.indices = np.append(self.indices, idx,  axis=0)
        self.neighbours += neighbour_idx
        self.shape[0] += n
        self.n = np.prod(self.shape) * self.n_base

    # =========================================================================

    def equivalent(self, other):
        """ Check if other lattice has same structure (atoms and their positions)

        Parameters
        ----------
        other: Lattice

        Returns
        -------
        equivalent: bool
        """
        if len(self.atoms) != len(other.atoms):
            return False
        unmatched_atoms = list(self.atoms)
        unmatched_pos = list(self.atom_positions)
        for i in range(self.n_base):
            try:
                unmatched_atoms.remove(other.atoms[i])
                unmatched_pos.remove(other.atom_positions[i])
            except ValueError:
                return False
        return (not unmatched_atoms) and (not unmatched_pos)

    def __str__(self):
        name = "".join(self.atoms)
        string = name + "-Lattice:\n"
        for i in range(self.n_base):
            atom = f"'{self.atoms[i]}'"
            string += f"   {i+1}: {atom:<5} @ {self.atom_positions[i]}\n"
        string += f"   Distances: " + ", ".join([f"{d}" for d in self.distances])
        return string + "\n"

    def show(self, show=True):
        """ Plot the cached lattice

        Parameters
        ----------
        show: bool, optional
            parameter for pyplot

        Returns
        -------
        plot: LatticePlot
        """
        if self.n == 0:
            print("[ERROR] Build lattice before plotting!")
            return None
        offset = 1
        hop_cols = ["0.2", "0.4", "0.6"]
        plot = LatticePlot(self.dim)
        for i in range(self.n):
            n, alpha = self.get(i)
            pos = self.get_position(n, alpha)
            plot.draw_site(self.get_atom(alpha), pos)
            neighbours = self.neighbours[i]
            for i_hop in range(len(self.distances)):
                for j in neighbours[i_hop]:
                    pos2 = self.position(j)
                    plot.draw_line([pos, pos2], color=hop_cols[i_hop])
        plot.rescale(offset)
        if show:
            plot.show()
        return plot


# =========================================================================
# 2D lattice prefabs
# =========================================================================


def square_lattice(shape=(1, 1), name="A", a=1.):
    """ square lattice prefab with one atom at the origin of the unit cell

    Parameters
    ----------
    shape: tuple, optional
        shape to build lattice, default: (1, 1)
        if None, the lattice won't be built on initialization
    name: str, optional
        name of the atom, default: "A"
    a: float, optional
        lattice constant, default: 1

    Returns
    -------
    latt: Lattice
    """
    latt = Lattice(np.eye(2)*a)
    latt.add_atom(name=name)
    latt.calculate_distances(1)
    if shape is not None:
        latt.build(shape)
    return latt

# =========================================================================
# 2D lattice prefabs
# =========================================================================


def cubic_lattice(shape=(1, 1, 1), name="A", a=1):
    """ cubic lattice prefab with one atom at the origin of the unit cell

    Parameters
    ----------
    shape: tuple, optional
        shape to build lattice, default: (1, 1, 1)
        if None, the lattice won't be built on initialization
    name: str, optional
        name of the atom, default: "A"
    a: float, optional
        lattice constant, default: 1

    Returns
    -------
    latt: Lattice
    """
    latt = Lattice(np.eye(3)*a)
    latt.add_atom(name=name)
    latt.calculate_distances(1)
    if shape is not None:
        latt.build(shape)
    return latt
