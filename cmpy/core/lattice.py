# -*- coding: utf-8 -*-
"""
Created on 26 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
import itertools
import numpy as np
from sciutils import vrange, distance, Plot
from matplotlib.collections import LineCollection


class AppendError(Exception):

    def __init__(self, msg):
        super().__init__("Can't append lattice: " + msg)


def iter_indices(n_vecs, n_alpha):
    """ Iterate over range of indices

    Parameters
    ----------
    n_vecs: array_like
        translation vectors
    n_alpha: array_like
        atom-indices

    Returns
    -------
    idx: np.ndarray
    """
    indices = list()
    for n in n_vecs:
        for alpha in range(n_alpha):
            indices.append((n, alpha))
    return indices


def index_array(n_vecs, n_alpha):
    """ Iterate over range of indices and store as single array

    Parameters
    ----------
    n_vecs: array_like
        translation vectors
    n_alpha: array_like
        atom-indices

    Returns
    -------
    idx: np.ndarray
    """
    indices = list()
    for n in n_vecs:
        for alpha in range(n_alpha):
            indices.append([*n, alpha])
    return np.array(indices)


def translate(vectors, n):
    """ Calculate the position from the main lattice coefficients

    Parameters
    ----------
    vectors: array_like
        lattice vectors
    n: array_like:
        lattice coefficients

    Returns
    -------
    position: np.ndarray
    """
    return np.asarray(vectors) @ n
    # return np.sum(np.asarray(n) * np.asarray(vectors), axis=1)


class LatticePlotBase(Plot):

    def __init__(self, size=10, color=True, lw=1.):
        super().__init__()
        self.atom_size = size
        self.color = color
        self.lw = lw

    def draw_sites(self, positions, label="", col=None):
        raise NotImplementedError()

    def draw_hoppings(self, segments, color="k"):
        coll = LineCollection(segments, color=color, lw=self.lw, zorder=1)
        self.ax.add_collection(coll)


class LatticePlot1D(LatticePlotBase):

    def __init__(self, size=10, color=True, lw=1.):
        super().__init__(size, color, lw)
        self.set_labels(xlabel="$x [a]$")
        self._limits = np.zeros(2)

    def draw_sites(self, positions, label="", col=None):
        x, y = positions.T
        xmin, xmax = np.min(x), np.max(x)
        if xmin < self._limits[0]:
            self._limits[0] = xmin
        if xmax > self._limits[1]:
            self._limits[1] = xmax
        col = "k" if not self.color else col
        self.scatter(x, y, zorder=10, s=self.atom_size, color=col, label=label)
        self.rescale()

    def draw_hoppings_cont(self, start, stop, color="k"):
        points = np.array([start, stop])
        self.plot(*points.T, color=color, lw=self.lw, zorder=1)

    def add_map(self, data, label=None, color=None):
        line = self.plot(*data, color=color)
        self.set_limits(ylim=0.1)
        if label:
            self.set_labels(ylabel=label)
        return line

    def rescale(self, offset=1.):
        xlim = self._limits + np.array([-offset, offset])
        self.set_limits(xlim)


class LatticePlot2D(LatticePlotBase):

    def __init__(self, size=10, color=True, lw=1.):
        super().__init__(size, color, lw)
        self.set_equal_aspect()
        self.set_labels(xlabel="$x [a]$", ylabel="$y [a]$")
        self._limits = np.zeros((2, 2))

    def draw_sites(self, positions, label="", col=None):
        positions = positions.T
        mins, maxs = np.min(positions, axis=1), np.max(positions, axis=1)
        idx = mins < self._limits[:, 0]
        self._limits[idx, 0] = mins[idx]
        idx = maxs > self._limits[:, 1]
        self._limits[idx, 1] = maxs[idx]

        col = "k" if not self.color else col
        self.scatter(*positions, zorder=10, s=self.atom_size, color=col, label=label)
        self.rescale()

    def add_map(self, data, cmap="Reds", colorbar=False, label=None):
        im = self.ax.contourf(*data, zorder=0, cmap=cmap)
        if colorbar:
            cb = self.colorbar(im, orientation="vertical")
            if label:
                cb.set_label(label)
        return im

    def rescale(self, offset=1.):
        delta = np.array([[-offset, offset], [-offset, offset]])
        self.set_limits(*self._limits + delta)


# =========================================================================
# LATTICE OBJECT
# =========================================================================


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

        # Cell data
        self.distances = list()
        self._base_neighbors = list()
        self.atoms = list()
        self.atom_positions = list()

        # Chached data if lattice is built
        self.shape = np.zeros(self.dim)
        self.indices = None
        self.neighbours = list()

    # 1D lattice prefabs

    @classmethod
    def chain(cls, size=1, name="A", a=1.):
        """ square lattice prefab with one atom at the origin of the unit cell

        Parameters
        ----------
        size: int, default: 1
            size of lattice chain, default: (1, 1)
            if None, the lattice won't be built on initialization
        name: str, optional
            name of the atom, default: "A"
        a: float, optional
            lattice constant, default: 1

        Returns
        -------
        latt: Lattice
        """
        shape = (size, 1)
        latt = cls(np.eye(2) * a)
        latt.add_atom(name=name)
        latt.calculate_distances(1)
        if shape is not None:
            latt.build(shape)
        return latt

    # 2D lattice prefabs

    @classmethod
    def square(cls, shape=(1, 1), name="A", a=1.):
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
        latt = cls(np.eye(2) * a)
        latt.add_atom(name=name)
        latt.calculate_distances(1)
        if shape is not None:
            latt.build(shape)
        return latt

    @classmethod
    def hexagonal(cls, shape=(2, 1), atom1="A", atom2="B", a=1.):
        vectors = a * np.array([[np.sqrt(3), np.sqrt(3) / 2],
                                [0, 3 / 2]])
        latt = cls(vectors)
        latt.add_atom(atom1)
        latt.add_atom(atom2, pos=[0, a])
        latt.calculate_distances(1)
        if shape is not None:
            latt.build_rect(*shape)
        return latt

    # 3D lattice prefabs

    @classmethod
    def cubic(cls, shape=(1, 1, 1), name="A", a=1):
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
        latt = cls(np.eye(3) * a)
        latt.add_atom(name=name)
        latt.calculate_distances(1)
        if shape is not None:
            latt.build(shape)
        return latt

    # =========================================================================

    def real_dim(self):
        dim = self.dim
        y = set()
        for i in range(self.n):
            pos = self.position(i)
            y.add(pos[1])
        if len(list(y)) == 1:
            dim = 1
        return dim

    @property
    def n_base(self):
        """ int: number of sites in unit-cell"""
        return len(self.atoms)

    @property
    def n_dist(self):
        """ int: Number of precalculated distances"""
        return len(self.distances)

    def copy(self):
        """ Create new lattice with equivalent setup

        Returns
        -------
        latt: Lattice
        """
        latt = Lattice(self.vectors)
        latt.distances = self.distances
        latt.atoms = self.atoms
        latt.atom_positions = self.atom_positions

        latt.shape = self.shape
        latt.indices = self.indices
        latt.neighbours = self.neighbours
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
            raised if position is allready occupied.
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
        alpha = np.where((self.atom_positions == r_alpha).all(axis=1))[0][0]
        idx = n, alpha
        return idx

    def estimate_index(self, pos):
        """ Returns lattice index (n, alpha) for global position.

        Parameters
        ----------
        pos: array_like
            global site position.

        Returns
        -------
        n: np.ndarray
            estimated translation vector n
        """
        pos = np.asarray(pos)
        n = np.asarray(np.floor(pos @ np.linalg.inv(self.vectors.T)), dtype="int")
        return n

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

    def calculate_neighbours(self, n=None, alpha=0, dist_idx=0, array=False):
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
        for idx1 in self.neighbour_range(n, dist_idx):
            # if np.isclose(self.distance(idx, idx1), dist, atol=1e-5):
            if abs(self.distance(idx, idx1) - dist) < 1e-5:
                if array:
                    idx1 = [*idx1[0], idx1[1]]
                indices.append(idx1)
        return indices

    def get_neighbours(self, idx, dist_idx=0):
        """ Transform stored neighbour indices

        Parameters
        ----------
        idx: tuple
            lattice vector (n, alpha) of first site
        dist_idx: int, default
            index of distance to neighbours, defauzlt is 0 (nearest neighbours).
        """
        n, alpha = np.array(idx[:-1]), idx[-1]
        transformed = list()
        for idx in self._base_neighbors[alpha][dist_idx]:
            idx_t = idx.copy()
            idx_t[:-1] += n
            transformed.append(idx_t)
        return transformed

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
        """ Calculate n lowest distances between sites in bravais lattice and the neighbors of the cell

        Checks distances between all sites of the bravais lattice and saves n lowest values.
        The neighbor lattice-indices of the unit-cell are also stored for later use.
        This speeds up many calculations like finding nearest neighbours.

        Parameters
        ----------
        n: int, optional
            number of distances of lattice structure to calculate.
            The default is 1 (nearest neighbours).
        """
        # Calculate n lowest distances of lattice structure
        n += 1
        n_vecs = vrange(self.dim * [np.arange(-n, n)])
        r_vecs = [self.get_position(*idx) for idx in iter_indices(n_vecs, self.n_base)]
        pairs = list(itertools.product(r_vecs, self.atom_positions))
        distances = list({distance(r1, r2) for r1, r2 in pairs})
        distances.sort()
        self.distances = distances[1:n]

        # Calculate cell-neighbors.
        neighbours = list()
        for alpha in range(self.n_base):
            site_neighbours = list()
            for i_dist in range(len(self.distances)):
                # Get neighbour indices of site for distance level
                site_neighbours.append(self.calculate_neighbours(alpha=alpha, dist_idx=i_dist, array=True))
            neighbours.append(site_neighbours)
        self._base_neighbors = neighbours

    # =========================================================================

    @property
    def n_cells(self):
        return int(np.prod(self.shape))

    @property
    def n(self):
        return len(self.indices) if self.indices is not None else 0

    @property
    def slice_shape(self):
        """ np.ndarray: shape of slice of built lattice"""
        return self.shape[1:]

    @property
    def slice_cells(self):
        return int(np.prod(self.slice_shape))

    @property
    def slice_sites(self):
        """ int: number of sites in slice of built lattice"""
        return np.prod(self.slice_shape) * self.n_base

    def get(self, i):
        """ Get the lattice index of a site  with index i in the built lattice

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

    def get_list_idx(self, n, alpha=0):
        return (self.indices == [*n, alpha]).all(axis=1).nonzero()[0][0]

    def get_alpha(self, i):
        """ Get the atom index alpha of the given site

        Parameters
        ----------
        i: int
            index of chached site

        Returns
        -------
        alpha: int
        """
        idx = self.indices[i]
        return idx[-1]

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

    def position_array(self, split_atoms=False):
        """ Returns a list of all positions in the built lattice

        Parameters
        ----------
        split_atoms: bool, default: True
            Split position array according to atom type

        Returns
        -------
        positions: (M, N/M) or (N) np.nparray
            if split_atoms is True, the position array is a 2dimensional array with M different atoms
            and N/M positions. Otherwise, only the N positions are returned
        """
        n_cells = int(self.n / self.n_base)
        positions = np.asarray([self.position(i) for i in range(self.n)])
        if split_atoms:
            return positions.reshape((self.n_base, n_cells, self.dim), order="F")
        else:
            return positions

    def _cached_neighbours(self, i_site, site_idx=None, indices=None):
        """ Get indices of cached neighbors

        Parameters
        ----------
        i_site: int
            index of cached site
        site_idx: array_like, optional
            lattice index of site. Default:
            Use existing indices stored in instance
        indices: array_like, optional
            all chached lattice indices. Default:
            Use existing indices stored in instance

        Returns
        -------
        indices: list
        """
        if indices is None:
            indices = self.indices
        if site_idx is None:
            site_idx = indices[i_site]

        # Get relevant index range to only look for neighbours
        # in proximity of site (larger than highest distance)
        n_dist = len(self.distances)
        offset = int((n_dist + 1) * self.slice_sites)
        i0 = max(i_site - offset, 0)
        i1 = min(i_site + offset, self.n + len(indices))
        site_window = indices[i0:i1]

        # Get neighbour indices of site in proximity
        neighbour_indices = list()
        for i_dist in range(n_dist):
            # Get neighbour indices of site for distance level
            dist_neighbours = list()
            for idx in self.get_neighbours(site_idx, i_dist):
                # Find site of neighbour and store if in cache
                hop_idx = np.where(np.all(site_window == idx, axis=1))[0]
                if len(hop_idx):
                    dist_neighbours.append(hop_idx[0] + i0)
            neighbour_indices.append(dist_neighbours)
        return neighbour_indices

    def build_positions(self, n_vecs):
        indices = list()
        for n in n_vecs:
            for alpha in range(self.n_base):
                indices.append([*n, alpha])
        return np.array(indices)

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
            index of neighbours in cached site list
        """
        num_sites = len(n_vecs) * self.n_base

        # Build lattice indices from translation vectors
        indices = self.build_positions(n_vecs)

        # get all sites (cached and new)
        if self.indices is not None:
            all_sites = np.append(self.indices, indices, axis=0)
        else:
            all_sites = indices

        # Find neighbours of each site in the "indices" list
        neighbours = list()
        offset = self.n
        for i in range(num_sites):  # prange(num_sites, header=f"Analyzing lattice", enabled=num_sites >= 5000):
            idx = indices[i]
            site = i + offset
            # Get neighbour indices of site
            neighbour_indices = self._cached_neighbours(site, idx, all_sites)
            neighbours.append(neighbour_indices)
        return indices, neighbours

    def build(self, shape=None):
        """ Build cache of finite size lattice with given shape

        Parameters
        ----------
        shape: array_like
            shape of finite size lattice
        """
        self.indices = None
        self.neighbours = None
        self.shape = np.ones(self.dim, dtype="int") if shape is None else np.asarray(shape)

        n_vecs = vrange([range(s) for s in shape])
        self.indices, self.neighbours = self.build_section(n_vecs)

    def build_rect(self, width, height):
        """ Build a lattice in shape of a rectangle
              (0, h)          (w, h)
                x3______________x2
                 |              |
                 |              |
                x0--------------x1
              (0, 0)          (w, 0)

        Parameters
        ----------
        width: int
        height: int

        Returns
        -------
        """
        self.indices = None
        self.neighbours = None

        shape = np.array([width, height])

        n1 = self.estimate_index((width, 0))
        n2 = self.estimate_index((width, height))
        n3 = self.estimate_index((0, height))

        x, y = np.array([n1, n2, n3]).T
        offset = 2
        xrange = range(min(np.min(x), 0) - offset, max(x) + offset)
        yrange = range(min(np.min(y), 0) - offset, max(y) + offset)
        n_vecs = vrange([xrange, yrange])
        final_nvecs = list()
        for n in n_vecs:
            if self.check_nvec(n, width, height):
                final_nvecs.append(n)

        self.shape = shape
        self.indices, self.neighbours = self.build_section(final_nvecs)

    def check_nvec(self, n, width, height):
        x, y = self.get_position(n)
        if (x < 0) or (x > width):
            return False
        if (y < 0) or (y > height):
            return False
        return True

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
        new_indices, new_neighbours, = self.build_section(new_vecs)

        # build new section
        indices = np.append(self.indices, new_indices,  axis=0)
        neighbours = self.neighbours + new_neighbours

        # connect new and old section
        offset = int(len(self.distances) * self.slice_sites)
        i0, i1 = self.n - offset, self.n + offset
        for i in range(i0, i1):
            idx = indices[i]
            neighbours[i] = self._cached_neighbours(i, idx, indices)

        self.shape[0] += n
        self.indices = indices
        self.neighbours = neighbours

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
        only_x = (x is not None) and (y is None) and (z is None)
        if only_x:
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

    def iter_neighbor_indices(self, i):
        neighbours = self.neighbours[i]
        for distidx in range(len(neighbours)):
            for j in neighbours[distidx]:
                yield distidx, j

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

    def show(self, show=True, size=10., color=True, margins=1., show_hop=True, lw=1.):
        """ Plot the cached lattice

        Parameters
        ----------
        show: bool, default: True
            parameter for pyplot
        size: float, default: 10
            size of the atom marker
        color: bool, default: True
            If True use colors in the plot
        margins: float, default: 0.5
            Margins of the plot
        show_hop: bool, default: True
            Draw hopping connections if True
        lw: float, default: 1
            Line width of the hopping connections
        Returns
        -------
        plot: LatticePlot
        """
        if self.n == 0:
            print("[ERROR] Build lattice before plotting!")
            return None
        n_cells = int(self.n / self.n_base)
        positions = np.asarray([self.position(i) for i in range(self.n)])
        atom_positions = positions.reshape((self.n_base, n_cells, self.dim), order="F")

        segments = list()
        if show_hop:
            for i in range(self.n):
                neighbours = self.neighbours[i]
                for i_hop in range(len(self.distances)):
                    for j in neighbours[i_hop]:
                        if j > i:
                            segments.append([positions[i], positions[j]])

        dim = self.real_dim()
        # 1D Plotting
        if dim == 1:
            plot = LatticePlot1D(size=size, color=color, lw=lw)
            for i_at in range(self.n_base):
                plot.draw_sites(atom_positions[i_at])
            if show_hop:
                plot.draw_hoppings_cont(positions[0], positions[-1])
        # 2D Plotting
        elif dim == 2:
            plot = LatticePlot2D(size=size, color=color, lw=lw)
            for i_at in range(self.n_base):
                plot.draw_sites(atom_positions[i_at])
            if show_hop:
                plot.draw_hoppings(segments)
        else:
            raise NotImplementedError()

        plot.rescale(margins)
        if show:
            plot.show()
        return plot
