# -*- coding: utf-8 -*-
"""
Created on 26 Mar 2019
author: Dylan Jones

project: LatticeQMC
version: 1.0
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Golden ratio as standard ratio for plot-figures
GOLDEN_RATIO = (np.sqrt(5) - 1.0) / 2.0


class AppendError(Exception):

    def __init__(self, msg):
        super().__init__("Can't append lattice: " + msg)


def pts_to_inch(pts):
    return pts * (1. / 72.27)


def get_figsize(width=None, height=None, ratio=None):
    # Width and height
    if (width is not None) and (height is not None):
        width = pts_to_inch(width)
        height = pts_to_inch(height)
    else:
        if ratio is None:
            ratio = GOLDEN_RATIO
        # Width and ratio
        if width is not None:
            width = pts_to_inch(width)
            height = width * ratio
        # height and ratio
        elif height is not None:
            height = pts_to_inch(height)
            width = height / ratio
    return width, height


class LatticePlotBase:

    def __init__(self, size=10, color=True, lw=1., proj=None):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection=proj)
        self.radius = size
        self.atom_size = size ** 2
        self.color = color
        self.lw = lw

    def set_figsize(self, width=None, height=None, ratio=None):
        if width is None and height is None and ratio is None:
            return
        width, height = get_figsize(width, height, ratio)
        self.fig.set_size_inches(width, height)

    def set_equal_aspect(self):
        self.ax.set_aspect("equal", "box")

    def colorbar(self, im, *args, orientation="vertical", **kwargs):
        divider = make_axes_locatable(self.ax)
        if orientation == "vertical":
            cax = divider.append_axes("right", size="5%", pad=0.05)
        elif orientation == "horizontal":
            cax = divider.append_axes("bottom", size="5%", pad=0.6)
        else:
            allowed = ["vertical", "horizontal"]
            raise ValueError(f"Invalid orientation: {orientation}. Must be in {allowed}")
        return self.fig.colorbar(im, ax=self.ax, cax=cax, orientation=orientation, *args, **kwargs)

    def draw_sites(self, positions, label="", col=None):
        raise NotImplementedError()

    def draw_hoppings(self, segments, color="k"):
        coll = LineCollection(segments, color=color, lw=self.lw, zorder=1)
        self.ax.add_collection(coll)

    def print_indices(self, positions, offset=0.1):
        offset = np.ones_like(positions[0]) * offset
        for i, pos in enumerate(positions):
            lowerleft = np.asarray(pos) + offset
            self.ax.text(*lowerleft, s=str(i), va="bottom", ha="left")

    def tight(self, *args, **kwargs):
        self.fig.tight_layout(*args, **kwargs)

    def show(self, tight=True):
        if tight:
            self.tight()
        plt.show()


class LatticePlot1D(LatticePlotBase):

    def __init__(self, size=10, color=True, lw=1.):
        super().__init__(size, color, lw)
        self.ax.set_xlabel("$x [a]$")
        self._limits = np.zeros(2)

    def draw_sites(self, positions, label="", col=None):
        x, y = positions.T
        xmin, xmax = np.min(x), np.max(x)
        if xmin < self._limits[0]:
            self._limits[0] = xmin
        if xmax > self._limits[1]:
            self._limits[1] = xmax
        col = "k" if not self.color else col
        self.ax.scatter(x, y, zorder=10, s=self.atom_size, color=col, label=label)
        self.rescale()

    def draw_hoppings_cont(self, start, stop, color="k"):
        points = np.array([start, stop])
        self.ax.plot(*points.T, color=color, lw=self.lw, zorder=1)

    def add_map(self, data, label=None, color=None):
        line = self.ax.plot(*data, color=color)
        self.ax.set_ylim(ylim=0.1)
        if label:
            self.ax.set_ylabel(label)
        return line

    def rescale(self, offset=1.):
        xlim = self._limits + np.array([-offset, offset])
        self.ax.set_xlim(xlim)


class LatticePlot2D(LatticePlotBase):

    def __init__(self, size=10, color=True, lw=1.):
        super().__init__(size, color, lw)
        self.set_equal_aspect()
        self.ax.set_xlabel("$x [a]$")
        self.ax.set_ylabel("$y [a]$")
        self._limits = np.zeros((2, 2))

    def draw_sites(self, positions, label="", col=None):
        positions = positions.T
        mins, maxs = np.min(positions, axis=1), np.max(positions, axis=1)
        idx = mins < self._limits[:, 0]
        self._limits[idx, 0] = mins[idx]
        idx = maxs > self._limits[:, 1]
        self._limits[idx, 1] = maxs[idx]

        col = "k" if not self.color else col
        self.ax.scatter(*positions, zorder=10, s=self.atom_size, color=col, label=label)
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
        xlim, ylim = self._limits + delta
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)


class LatticePlot3D(LatticePlotBase):

    def __init__(self, size=10, color=True, lw=1.):
        super().__init__(size, color, lw, proj="3d")
        self.set_equal_aspect()
        self.ax.set_xlabel("$x$")
        self.ax.set_ylabel("$y$")
        self.ax.set_ylabel("$z$")
        self._limits = np.zeros((3, 2))

    def draw_sites(self, positions, label="", col=None):
        positions = positions.T
        mins, maxs = np.min(positions, axis=1), np.max(positions, axis=1)
        idx = mins < self._limits[:, 0]
        self._limits[idx, 0] = mins[idx]
        idx = maxs > self._limits[:, 1]
        self._limits[idx, 1] = maxs[idx]

        col = "k" if not self.color else col
        self.ax.scatter(*positions, zorder=3, s=self.atom_size, color=col, label=label, alpha=1)
        self.rescale()

    def draw_hoppings(self, segments, color="k"):
        coll = Line3DCollection(segments, color=color, lw=self.lw, zorder=1)
        self.ax.add_collection(coll)

    def add_map(self, data, cmap="Reds", colorbar=False, label=None):
        im = self.ax.contourf(*data, zorder=0, cmap=cmap)
        if colorbar:
            cb = self.colorbar(im, orientation="vertical")
            if label:
                cb.set_label(label)
        return im

    def rescale(self, offset=1.):
        delta = np.array([[-offset, offset], [-offset, offset], [-offset, offset]])
        xlim, ylim, zlim = self._limits + delta
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_ylim(zlim)


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
    r""" Calculate the position from the main lattice coefficients.

    ..math::

        \mathbf{R} = \sum_i^d n_i \mathbf{v}_i

    Parameters
    ----------
    vectors: (N, N) ndarray
        Lattice vectors
    n: (N) ndarray
        Lattice coefficients.

    Returns
    -------
    position: (N) ndarray
    """
    return np.asarray(vectors) @ n
    # return np.sum(np.asarray(n) * np.asarray(vectors), axis=1)


def vrange(axis_ranges):
    """ Return evenly spaced vectors within a given interval.

    Parameters
    ----------
    axis_ranges: array_like
        ranges for each axis.

    Returns
    -------
    vectors: np.ndarray
    """
    axis = np.meshgrid(*axis_ranges)
    grid = np.asarray([np.asarray(a).flatten("F") for a in axis]).T
    n_vecs = list(grid)
    n_vecs.sort(key=lambda x: x[0])
    return n_vecs


def distance(r1, r2):
    """ Calculates the euclidian distance bewteen two points.

    Parameters
    ----------
    r1: (N) ndarray
        First input point.
    r2: (N) ndarray
        Second input point of matching size.

    Returns
    -------
    distance: float
    """
    return np.sqrt(np.sum((r1 - r2)**2))


def split_idx(idx):
    """ Splits the full index array [n_1, ..., n_n, alpha] into the two indices

    Parameters
    ----------
    idx: array_like
        Full index array

    Returns
    -------
    n: array_like
    alpha: int
    """
    return idx[:-1], idx[-1]


# =========================================================================
# LATTICE OBJECT
# =========================================================================


class Lattice:

    DIST_DECIMALS = 5

    def __init__(self, vectors):
        """ Initialize general bravais Lattice

        Parameters
        ----------
        vectors: (N, N) ndarray
            Primitive vectors of lattice.
        """
        self.vectors = np.asarray(vectors)
        self.dim = len(vectors)

        # Cell data
        self.distances = list()
        self._base_neighbors = list()
        self.atoms = list()
        self.atom_positions = list()

        # Lattice Cache
        self.n_sites = 0
        self.shape = None
        self.indices = None
        self.neighbours = None

    @classmethod
    def square_cell(cls, a=1.):
        """ Creates a lattice with square primitive vectors."""
        return cls(np.eye(2) * a)

    @classmethod
    def hexagonal_cell(cls, a=1.):
        """ Creates a lattice with hexagonal primitive vectors."""
        vectors = a * np.array([[np.sqrt(3), np.sqrt(3) / 2], [0, 3 / 2]])
        return cls(vectors)

    @classmethod
    def cubic_cell(cls, a=1.):
        """ Creates a lattice with cubic primitive vectors."""
        return cls(np.eye(3) * a)

    @classmethod
    def square(cls, name="A", a=1., neighbour_dist=1, shape=None):
        """ Simple-square lattice prefab with one atom at the origin of the unit cell. """
        latt = cls(np.eye(2) * a)
        latt.add_atom(name=name)
        latt.calculate_distances(neighbour_dist)
        if shape is not None:
            latt.build(shape)
        return latt

    @classmethod
    def hexagonal(cls, atom1="A", atom2="B", a=1., neighbour_dist=1, shape=None):
        """ Hexagonal lattice prefab with otwo atoms in the unit cell. """
        vectors = a * np.array([[np.sqrt(3), np.sqrt(3) / 2],
                                [0, 3 / 2]])
        latt = cls(vectors)
        latt.add_atom(atom1)
        latt.add_atom(atom2, pos=[0, a])
        latt.calculate_distances(neighbour_dist)
        if shape is not None:
            latt.build(shape)
        return latt

    @classmethod
    def sc(cls, name="A", a=1., neighbour_dist=1, shape=None):
        """ Simple-cubic lattice."""
        latt = cls.cubic_cell(a)
        latt.add_atom(name=name)
        latt.calculate_distances(neighbour_dist)
        if shape is not None:
            latt.build(shape)
        return latt

    @classmethod
    def bcc(cls, name1="A", name2=None, a=1., neighbour_dist=1, shape=None):
        """ Body-centered-cubic lattice."""
        latt = cls.cubic_cell(a)
        name2 = name2 or name1
        latt.add_atom(name1, pos=(0, 0, 0))
        latt.add_atom(name2, pos=np.ones(3) * a / 2)
        latt.calculate_distances(neighbour_dist)
        if shape is not None:
            latt.build(shape)
        return latt

    # =========================================================================

    @property
    def n_base(self):
        """ int: number of sites in unit-cell"""
        return len(self.atoms)

    @property
    def n_dist(self):
        """ int: Number of precalculated distances"""
        return len(self.distances)

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

    # =========================================================================

    def add_atom(self, name="A", pos=None, neighbour_dist=0):
        """ Add site to lattice cell.

        Parameters
        ----------
        name: str, optional
            Name of the site.
        pos: array_like, optional
            position of site in the lattice cell.
        neighbour_dist: int, optional
            The number of neighbor distance to calculate. The default is 0.
            If not calculated, this has to be done manually after configuring lattice.

        Raises
        ------
        ValueError:
            raised if position is allready occupied.
        """
        if pos is None:
            pos = np.zeros(self.vectors.shape[0])
        else:
            pos = np.asarray(pos)
        if any(np.all(pos == x) for x in self.atom_positions):
            raise ValueError(f"Position {pos} allready occupied")
        self.atoms.append(name)
        self.atom_positions.append(pos)
        if neighbour_dist:
            self.calculate_distances(neighbour_dist)

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

    def get_latt_index(self, pos):
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
        n: (N) array_like
            translation vector.
        alpha: int, optional
            site index, default is 0.
        Returns
        -------
        pos: (N) np.ndarray
        """
        r = self.atom_positions[alpha]
        return r + translate(self.vectors, n)

    def translate_cell(self, n):
        """ Translate all contents of the unit cell

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

    def _neighbour_range(self, n=None, cell_range=1):
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
        for idx1 in self._neighbour_range(n, dist_idx):
            # if np.isclose(self.distance(idx, idx1), dist, atol=1e-5):
            if np.round(abs(self.distance(idx, idx1) - dist), decimals=self.DIST_DECIMALS) == 0.0:
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
        for idx1 in self._neighbour_range(n, dist_idx + 1):
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
        distances = list(set(np.round([distance(r1, r2) for r1, r2 in pairs], decimals=self.DIST_DECIMALS)))
        distances.sort()
        distances.remove(0.0)
        self.distances = distances[0:n-1]

        # Calculate cell-neighbors.
        neighbours = list()
        for alpha in range(self.n_base):
            site_neighbours = list()
            for i_dist in range(len(self.distances)):
                # Get neighbour indices of site for distance level
                site_neighbours.append(self.calculate_neighbours(alpha=alpha, dist_idx=i_dist, array=True))
            neighbours.append(site_neighbours)
        self._base_neighbors = neighbours

    def cell_volume(self):
        if self.dim == 3:
            cross = np.cross(self.vectors[1], self.vectors[2])
            v = np.dot(self.vectors[0], cross)
        else:
            v = np.cross(self.vectors[0], self.vectors[1])
        return abs(v)

    # =========================================================================
    # CACHED LATTICE METHODS
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

    def get_list_idx(self, n, alpha=0):
        return (self.indices == [*n, alpha]).all(axis=1).nonzero()[0][0]

    def lattice_index(self, i):
        return split_idx(self.indices[i])

    def alpha(self, i):
        return self.indices[i][-1]

    def position(self, i):
        return self.get_position(*self.lattice_index(i))

    def dist_neighbours(self, i, dist=1):
        return self.neighbours[i][dist-1]

    def iter_neighbours(self, i):
        for i_dist in range(self.n_dist):
            for j in self.dist_neighbours(i, i_dist+1):
                yield i_dist, j

    def nearest_neighbours(self, i):
        return self.neighbours[i][0]

    def _find_neighbours(self, i_site, site_idx, indices, offset=None):
        """ Get indices of neighbors

        Parameters
        ----------
        i_site: int
            index of cached site
        site_idx: array_like, optional
            lattice index of site.
        indices: array_like, optional
            All new site indices.
        offset: int, optional
            Index offset for searching neighbors. Default is all.


        Returns
        -------
        indices: list
        """
        # Get relevant index range to only look for neighbours
        # in proximity of site (larger than highest distance)
        n_dist = len(self.distances)
        # offset = int((n_dist + 1) * self.slice_sites)
        offset = len(indices) if offset is None else offset

        i0 = max(i_site - offset, 0)
        i1 = min(i_site + offset, len(indices))
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

    def build_nvecs(self, width, height):
        """ Build a lattice in shape of a rectangle

        Parameters
        ----------
        width: int
        height: int

        Returns
        -------
        nvecs: array_like
        """
        width -= 1
        height -= 1

        n1 = self.estimate_index((width, 0))
        n2 = self.estimate_index((width, height))
        n3 = self.estimate_index((0, height))
        x, y = np.array([n1, n2, n3]).T
        offset = 1
        x0, x1 = min(np.min(x), 0) - offset, max(x) + offset
        y0, y1 = min(np.min(y), 0) - offset, max(y) + offset
        xrange = range(x0, x1)
        yrange = range(y0, y1)
        n_vecs = vrange([xrange, yrange])
        final_nvecs = list()
        for n in n_vecs:
            add = True
            for alpha in range(self.n_base):
                if not self._check_site(n, alpha, width, height):
                    add = False
                    break
            if add:
                final_nvecs.append(n)
        return final_nvecs

    def _build_section(self, n_vecs, indices=None, neighbor_window=None):
        num_sites = len(n_vecs) * self.n_base
        # Build lattice indices from translation vectors
        new_indices = list()
        for n in n_vecs:
            for alpha in range(self.n_base):
                new_indices.append([*n, alpha])
        new_indices = np.array(new_indices)
        # get all sites (cached and new)
        if indices is not None:
            all_indices = np.append(indices, new_indices, axis=0)
        else:
            all_indices = new_indices
        # Find neighbours of each site in the "new_indices" list
        new_neighbours = list()
        offset = len(indices) if indices is not None else 0
        for i in range(num_sites):  # prange(num_sites, header=f"Analyzing lattice", enabled=num_sites >= 5000):
            idx = new_indices[i]
            site = i + offset
            # Get neighbour indices of site
            neighbour_indices = self._find_neighbours(site, idx, all_indices, neighbor_window)
            new_neighbours.append(neighbour_indices)
        return new_indices, new_neighbours

    def _set_chache(self, shape, indices, neighbours):
        self.shape = shape
        self.n_sites = len(indices)
        self.indices = indices
        self.neighbours = neighbours

    def _build(self, shape):
        """ Build cache of finite size lattice with given shape

        Parameters
        ----------
        shape: array_like
            shape of finite size lattice
        """
        shape = np.asarray(shape)
        n_vecs = vrange([range(s) for s in shape])

        slice_sites = np.prod(shape[1:]) * self.n_base
        window = int((len(self.distances) + 1) * slice_sites)
        indices, neighbours = self._build_section(n_vecs, neighbor_window=window)
        return shape, indices, neighbours

    def build(self, shape):
        """ Build cache of finite size lattice with given shape and store it

        Parameters
        ----------
        shape: array_like
            shape of finite size lattice
        """
        shape, indices, neighbours = self._build(shape)
        self._set_chache(shape, indices, neighbours)
        return indices, shape

    def _check_site(self, n, alpha, width, height):
        x, y = self.get_position(n, alpha)
        if (x < 0) or (x > width):
            return False
        if (y < 0) or (y > height):
            return False
        return True

    def _build_rect(self, width, height):
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
        shape = np.array([width, height])
        nvecs = self.build_nvecs(width, height)
        slice_sites = np.prod(shape[1:]) * self.n_base
        window = int((len(self.distances) + 1) * slice_sites)
        indices, neighbours = self._build_section(nvecs, neighbor_window=window)
        return shape, indices, neighbours

    def build_rect(self, width, height):
        """ Build a lattice in shape of a rectangle and store it.
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
        shape, indices, neighbours = self._build_rect(width, height)
        self._set_chache(shape, indices, neighbours)
        return indices, neighbours

    def add_slices(self, n):
        """ Add n slices of allready built lattice cache

        Parameters
        ----------
        n: int
            number of slices to add
        """
        indices = self.indices
        neighbours = self.neighbours
        shape = self.shape
        n_old = len(indices)
        slice_shape = shape[1:]
        slice_sites = np.prod(slice_shape) * self.n_base
        new_xrange = shape[0] + np.arange(n)
        ranges = [new_xrange] + [range(s) for s in slice_shape]
        new_vecs = vrange(ranges)
        window = int((len(self.distances) + 1) * slice_sites)
        new_indices, new_neighbours, = self._build_section(new_vecs, indices, window)

        # build new section
        indices = np.append(indices, new_indices,  axis=0)
        neighbours = list(neighbours) + list(new_neighbours)

        # connect new and old section
        offset = int(len(self.distances) * slice_sites)
        i0, i1 = n_old - offset, n_old + offset
        for i in range(i0, i1):
            idx = indices[i]
            neighbours[i] = self._find_neighbours(i, idx, indices, window)

        shape[0] += n
        self._set_chache(shape, indices, neighbours)
        return indices, neighbours

    def set_periodic_boundary(self, axis=0):
        """ Adds the indices of the neighbours cycled around the given axis.

        Notes
        -----
        The lattice has to be buildt before applying the periodic boundarie conditions.
        Also the lattice has to be at least three atoms big in the specified directions.

        Parameters
        ----------
        axis: int or (N) array_like, optional
            One or multiple axises to apply the periodic boundary conditions.
            The default is the x-direction.
        """
        axis = np.atleast_1d(axis)
        for ax in axis:
            # Check the maximal distance of cells along the axis
            celldist_max = 0.0
            for i in range(self.n_sites):
                nx1 = self.indices[i, ax]
                for j in range(i + 1, self.n_sites):
                    nx2 = self.indices[j, ax]
                    delta = float(nx2 - nx1)
                    if delta > celldist_max:
                        celldist_max = delta

            # Add the cycling neighbour indices to the chached list.
            for i_dist in range(self.n_dist):
                for i in range(self.n_sites):
                    pos1 = self.position(i)
                    for j in range(i + 1, self.n_sites):
                        pos2 = self.position(j)
                        dist_offset = 0 if i_dist == 0 else self.distances[i_dist - 1]
                        dist = np.round(distance(pos1, pos2), decimals=self.DIST_DECIMALS)
                        if dist == celldist_max + dist_offset:
                            diff = pos2 - pos1
                            maxdist_axis = np.where(diff == np.amax(np.abs(diff)))[0]
                            if maxdist_axis == ax:
                                self.neighbours[i][i_dist].append(j)
                                self.neighbours[j][i_dist].append(i)

    def show(self, show=True, size=10., color=True, margins=1., show_hop=True, lw=1., show_indices=False):
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
        show_indices: bool, optional
            If 'True' the index of the sites will be shown.

        Returns
        -------
        plot: LatticePlot
        """
        if self.indices is None:
            dims = np.sum(self.vectors, axis=0)
            shape, indices, neighbours = self._build_rect(4*dims[0], 2*dims[1])
        else:
            indices = self.indices
            neighbours = self.neighbours

        n = len(indices)
        n_cells = int(n / self.n_base)
        positions = np.asarray([self.get_position(*split_idx(idx)) for idx in indices])
        atom_positions = positions.reshape((self.n_base, n_cells, self.dim), order="F")

        segments = list()
        if show_hop:
            for i in range(n):
                neighbor_list = neighbours[i]
                for i_hop in range(len(self.distances)):
                    for j in neighbor_list[i_hop]:
                        if j > i:
                            segments.append([positions[i], positions[j]])

        dim = self.dim
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
        elif dim == 3:
            plot = LatticePlot3D(size=size, color=color, lw=lw)
            for i_at in range(self.n_base):
                plot.draw_sites(atom_positions[i_at])
            if show_hop:
                plot.draw_hoppings(segments)
        else:
            raise NotImplementedError()

        if show_indices:
            positions = [self.position(i) for i in range(self.n_sites)]
            plot.print_indices(positions)

        plot.rescale(margins)
        if show:
            plot.show()
        return plot
