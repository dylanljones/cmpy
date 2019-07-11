# -*- coding: utf-8 -*-
"""
Created on  06 2019
author: dylan

project: cmpy2
version: 1.0
"""
import numpy as np
from scipy import interpolate
from sciutils import distance, chain, vlinspace, normalize, eta
from sciutils.terminal import Progress
from .lattice import Lattice
from .hamiltonian import TbHamiltonian
from .utils import HamiltonianCache, plot_bands, uniform_eye


def occupation_map(lattice, occ, margins=0.5, upsample=None):
    """ Builds grid-data from the occupation of the lattice

    Parameters
    ----------
    lattice: Lattice
        Lattice instance of the system
    occ: array_like
        Array of the occupation values of the system in the same order as the lattice sites
    margins: float, default: 0.5
        Margins of the generated grid
    upsample: float, default: None
        Upsample factor for the grid. If not specified, original number of samples is used

    Returns
    -------
    griddata: tuple of np.ndarray
    """
    positions = np.array([lattice.position(i) for i in range(len(occ))])
    x, y = positions.T
    if len(set(y)) == 1:
        if upsample:
            n_x = np.array(lattice.shape[0]) * upsample
            x_int = np.linspace(np.min(x), np.max(x), n_x)
            f = interpolate.interp1d(x, occ)
            return x_int, f(x_int)
        else:
            return x, occ
    else:
        if upsample is None:
            upsample = 1
            method = "nearest"
        else:
            method = "cubic"

        # Expand data
        n_x, n_y = np.array(lattice.shape) * upsample
        x_int = np.linspace(np.min(x) - margins, np.max(x) + margins, n_x)
        y_int = np.linspace(np.min(y) - margins, np.max(y) + margins, n_y)
        # Generate griddata
        xx, yy = np.meshgrid(x_int, y_int)
        zz = interpolate.griddata(positions, occ, (xx, yy), method=method)
        return xx, yy, zz

# =========================================================================
# TIGHT-BINDING MODEL
# =========================================================================


class TightBinding:

    ORBITALS = "s", "p_x", "p_y", "p_z"

    def __init__(self, vectors=np.eye(2), lattice=None):
        self.lattice = Lattice(vectors) if lattice is None else lattice
        self.n_orbs = None

        self.energies = list()
        self.hoppings = list()

        self.w_eps = 0
        self._disorder_onsite = None

        # Cached hamiltonian objects for more performance
        self._slice_ham_cache = HamiltonianCache()
        self._slice_hop_cache = HamiltonianCache()
        self._ham_cache = HamiltonianCache()

    @classmethod
    def square(cls, shape=(2, 1), eps=0., t=1., basis=None, name="A", a=1.):
        """ square device prefab with one atom at the origin of the unit cell

        Parameters
        ----------
        shape: tuple, optional
            shape to build lattice, the default is (1, 1)
            if None, the lattice won't be built on initialization
        eps: float, optional
            energy of the atom, the default is 0
        t: float, otional
            hopping parameter, the default is 1.
        basis: Basis, optional
            energy basis of one atom, if not None, use this instead
            of eps and t
        name: str, optional
            name of the atom, the default is "A"
        a: float, optional
            lattice constant, default: 1

        Returns
        -------
        model: TightBinding
        """
        self = cls(a * np.eye(2))
        if basis is not None:
            eps = basis.eps
            t = basis.hop
        self.add_atom(name, energy=eps)
        self.set_hopping(t)
        if shape is not None:
            self.build(shape)
        return self

    # =========================================================================

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
    def n_slices(self):
        return int(self.n_elements / self.slice_elements)

    @property
    def channels(self):
        """int: number of transmission channels of the system"""
        return np.prod(self.lattice.shape)

    @property
    def atoms(self):
        return self.lattice.atoms

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
        hoppings: iterable
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
        self._slice_ham_cache.clear()
        self._slice_hop_cache.clear()

    def clear_ham_cache(self):
        self._ham_cache.clear()
        # self._cached_ham = None

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
        self.clear_ham_cache()
        self.lattice.reshape(x, y, z)
        if (y is not None) or (z is not None):
            self.clear_slice_cache()
        self.shuffle()

    def set_disorder(self, w_eps):
        """ Set the amoount of disorder of the on-site energies

        Parameters
        ----------
        w_eps: float
            disorder amount
        """
        self.w_eps = w_eps
        self.shuffle()

    def onsite_disorder(self, i=None):
        if i is None:
            return self._disorder_onsite
        else:
            i, j = np.array([i, i+1]) * self.slice_elements
            return self._disorder_onsite[i:j, i:j]

    def shuffle(self):
        if self.w_eps:
            size = self.n_orbs * self.lattice.n
            self._disorder_onsite = uniform_eye(self.w_eps, size)

    # =========================================================================
    # Hamiltonians
    # =========================================================================

    def slice_hamiltonian(self):
        """ Get the slice hamiltonian of the model

        Notes
        -----
        Uses cached data if available, otherwise cache will be initialized

        Returns
        -------
        ham: Hamiltonian
        """
        if not self._slice_ham_cache:
            # Initialize slice Hamiltonian
            # ----------------------------
            n = self.lattice.slice_sites
            ham = TbHamiltonian.zeros(n, self.n_orbs, "complex")
            for i in range(n):
                _, alpha = self.lattice.get(i)
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

            self._slice_ham_cache.load(ham)
        else:
            # Reset slice Hamiltonian
            # -----------------------
            self._slice_ham_cache.reset()

        return self._slice_ham_cache.read()

    def get_slice_hamiltonian(self, i):
        """ Get the i-th slice hamiltonian of the model and add disorder

        Notes
        -----
        Uses cached data if available, otherwise cache will be initialized

        Parameters
        ----------
        i: int
            Index of the slice to return
        Returns
        -------
        ham: Hamiltonian
        """
        ham = self.slice_hamiltonian()
        # Apply Disorder
        if self.w_eps:
            return ham + self.onsite_disorder(i)
        else:
            return ham

    def slice_iterator(self):
        for i in range(self.n_slices):
            yield self.get_slice_hamiltonian(i)

    def slice_hopping(self):
        """ Get the hopping hamiltonian between two slices of the model

        Notes
        -----
        Uses cached data if available, otherwise cache will be initialized

        Returns
        -------
        ham: Hamiltonian
        """
        if not self._slice_hop_cache:
            # Initialize slice-hopping Hamiltonian
            # ------------------------------------
            n = self.lattice.slice_sites
            ham = TbHamiltonian.zeros(n, self.n_orbs, "complex")
            for i in range(n):
                # Hopping energies
                for dist in range(len(self.lattice.distances)):
                    for j in range(n):
                        if j + n in self.lattice.neighbours[i][dist]:
                            t = self.hoppings[dist]
                            ham.set(i, j, t)
            self._slice_hop_cache.load(ham)

        return self._slice_hop_cache.read()

    def hamiltonian(self, w_eps=0., blocksize=None):
        """ Get the full hamiltonian of the model and add disorder

        Notes
        -----
        Uses cached data if available, otherwise cache will be initialized

        Parameters
        ----------
        w_eps: float, optional
            on-site disorder strength
        blocksize: int, optional
            blocksize that the hamiltonian will be configured

        Returns
        -------
        ham: Hamiltonian
        """
        if not self._ham_cache:
            # Initialize Hamiltonian
            # ----------------------------
            n = self.lattice.n
            ham = TbHamiltonian.zeros(n, self.n_orbs, "complex")
            # energies = TbHamiltonian.zeros(n, self.n_orbs, "complex")
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
            self._ham_cache.load(ham)
        else:
            # Reset Hamiltonian
            # ----------------------------
            self._ham_cache.reset()

        if w_eps:
            self.set_disorder(w_eps)

        ham = self._ham_cache.read()
        if self.w_eps:
            return ham + self.onsite_disorder()
        else:
            return ham

    # =========================================================================
    # Properties
    # =========================================================================

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
        ham = TbHamiltonian.zeros(self.n_base, self.n_orbs, "complex")
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
        points: list of array_like
            k-points for the bandstructure
        n_e: int, default: 1000
            number of energy samples in each section
        thresh: float, default: 1e-9
            threshold to find energy-values
        scale: bool, default: True
            scale lengths of sections
        verbose: bool, default: True
            print progress if True

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

    def ldos(self, omegas, banded=False):
        ham = self.hamiltonian()
        return ham.ldos(omegas, banded)

    def dos(self, omegas, banded=False):
        return np.sum(self.ldos(omegas, banded), axis=1)

    def occupation(self, omega, banded=False):
        return normalize(self.ldos(omega, banded))

    def occupation_map(self, omega=eta, banded=True, margins=0.5, upsample=None):
        """ Builds the occupation map of the system

        Parameters
        ----------
        omega: complex, default: 0
            Energy of the system, default is e=0 (plus broadening)
        banded: bool, default: False
            Use the upper diagonal matrix for solving the eigenvalue problem. Full diagonalization is de
        margins: float, default: 0.5
            Margins of the generated grid
        upsample: float, default: None
            Upsample factor for the grid. If not specified, original number of samples is used

        Returns
        -------
        griddata: tuple of np.ndarray
        """
        occ = self.occupation(omega, banded=banded)
        return occupation_map(self.lattice, occ, margins, upsample)

    def inverse_participation_ratio(self, omega=eta):
        ldos = self.ldos(omega)
        return np.sum(np.power(ldos, 2)) / (np.sum(ldos) ** 2)

    # =========================================================================

    def plot_bands(self, points, pointnames, n_e=1000, thresh=1e-9, scale=True, verbose=True, show=True):
        """ Calculate and plot the band-structure in k-space between given k-points

        Parameters
        ----------
        points: list of array_like
            High symmetry k-points for the band-structure
        pointnames: list of str
            Names of the high symmetry k-points
        n_e: int, default: 1000
            number of energy samples in each section
        thresh: float, default: 1e-9
            threshold to find energy-values
        scale: bool, default: True
            scale lengths of sections
        verbose: bool, default: True
            print progress if True
        show: bool, default: True
            Show plot if True

        Returns
        -------
        plot: Plot
        """
        bands = self.bands(points, n_e, thresh, scale, verbose)
        return plot_bands(bands, pointnames, show)

    def show_occupation(self, omega=eta, banded=True, margins=0.5, upsample=None,
                        size=4, lw=0.5, show_hop=True, cmap="Reds", show=True, colorbar=True):

        psi_label = r"$\langle\Psi(x)\rangle$"
        occmap = self.occupation_map(omega, banded=banded, margins=margins, upsample=upsample)
        if len(occmap) == 2:
            # 1D Plot
            plot = self.show(False, size=size, lw=lw, margins=margins, show_hop=show_hop)
            plot.rescale(margins)
            plot.add_map(occmap, label=psi_label)
        else:
            # 2D Plot
            plot = self.show(False, color=False, size=size, lw=lw, margins=margins, show_hop=show_hop)
            plot.rescale(margins)
            plot.add_map(occmap, cmap=cmap, colorbar=colorbar, label=psi_label)

        plot.tight()
        if show:
            plot.show()
        return plot

    def show(self, show=True, *args, **kwargs):
        """ Plot the lattice of the device"""
        return self.lattice.show(show, *args, **kwargs)
