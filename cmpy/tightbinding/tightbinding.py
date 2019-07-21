# -*- coding: utf-8 -*-
"""
Created on  06 2019
author: dylan

project: cmpy2
version: 1.0
"""
import numpy as np
from scipy import interpolate
from sciutils import distance, chain, vlinspace, normalize, eta, List2D, blockmatrix_slices
from sciutils.terminal import Progress, prange
from cmpy.core.lattice import Lattice
from cmpy.core.hamiltonian import TbHamiltonian, Hamiltonian
from cmpy.core.utils import HamiltonianCache, plot_bands, uniform_eye

ORBITALS = "s", "p_x", "p_y", "p_z"

SPINS = "up", "down"
SOC_ORBS = ORBITALS[1:]

s_0 = np.zeros((2, 2))
# Pauli matrices
s_x = np.array([[0, 1], [1, 0]])
s_y = np.array([[0, -1j], [1j, 0]])
s_z = np.array([[1, 0], [0, -1]])

SOC = np.array([[s_0,     -1j*s_z, +1j*s_y],
                [+1j*s_z,     s_0, -1j*s_x],
                [-1j*s_y, +1j*s_x,     s_0]])

# =========================================================================
# TIGHT-BINDING BASIS
# =========================================================================


class State:

    def __init__(self, orb, spin=None):
        self.orb = orb
        self.spin = spin

    def is_orbit(self, orb):
        return self.orb == orb

    def is_spin(self, spin):
        return self.spin == spin

    def is_state(self, orb, spin):
        return self.is_orbit(orb) and self.is_spin(spin)

    def __str__(self):
        return f"{self.orb} {self.spin}"

    def __repr__(self):
        return f"State({self.orb}, {self.spin})"


def get_soc(s1, s2):
    if s1.orb == "s" or s2.orb == "s":
            return 0
    orb1, orb2 = SOC_ORBS.index(s1.orb), SOC_ORBS.index(s2.orb)
    s1, s2 = SPINS.index(s1.spin), SPINS.index(s2.spin)
    s = SOC[orb1, orb2]
    return s[s1, s2]


def get_atom_basis(energy=0., orbitals=None, spin=True):
    # Initialize energy- and orbit-array
    if not hasattr(energy, "__len__"):
        # Scalar energy
        if orbitals is None:
            orbitals = ORBITALS[0]
        n_orbs = len(orbitals)
        energy = np.ones(n_orbs) * energy
    else:
        # Energy array
        energy = np.asarray(energy)
        n_orbs = energy.shape[0] if len(energy.shape) else 1
        if orbitals is None:
            orbitals = ORBITALS[:n_orbs]
        elif len(orbitals) != n_orbs:
            raise ValueError(f"Size of energy-array doesn't match number of orbitals ({n_orbs}!={len(orbitals)})")

    # Initialize states
    if spin:
        n_orbs *= 2
        states = [State(orb, "up") for orb in orbitals]
        states += [State(orb, "down") for orb in orbitals]
        energy = np.append(energy, energy)
    else:
        states = [State(orb) for orb in orbitals]
    return states, np.eye(n_orbs) * energy


# =========================================================================
# METHODS
# =========================================================================


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

    def __init__(self, vectors=np.eye(2), lattice=None, spin=False):
        self.lattice = Lattice(vectors) if lattice is None else lattice
        self.spin = spin

        self.energies = list()
        self.basis_states = list()
        self.hopping = list()
        self.base_elements = 0

        self.w_eps = 0
        self._disorder_onsite = None

        # Cached hamiltonian objects for more performance
        self._slice_ham_cache = HamiltonianCache()
        self._slice_hop_cache = HamiltonianCache()
        self._ham_cache = HamiltonianCache()

    @classmethod
    def square(cls, shape=(2, 1), eps=0., t=1., name="A", a=1.):
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
        name: str, optional
            name of the atom, the default is "A"
        a: float, optional
            lattice constant, default: 1

        Returns
        -------
        model: cls
        """
        self = cls(a * np.eye(2))
        self.add_atom(name, energy=eps)
        self.set_hopping(t)
        if shape is not None:
            self.build(shape)
        return self

    @classmethod
    def chain(cls, size=1, eps=0., t=1., name="A", a=1.):
        return cls.square((size, 1), eps, t, name, a)

    @classmethod
    def hexagonal(cls, shape=(2, 1), eps1=0., eps2=0., t=1., atom1="A", atom2="B", a=1.):
        vectors = a * np.array([[np.sqrt(3), np.sqrt(3) / 2],
                                [0, 3 / 2]])
        self = cls(vectors)
        self.add_atom(atom1, energy=eps1)
        self.add_atom(atom2, energy=eps2, pos=[0, a])
        self.set_hopping(t)
        if shape is not None:
            self.build(shape)
        return self

    @classmethod
    def sc(cls, shape=(2, 1, 1), eps=0., t=1., name="A", a=1.):
        self = cls(np.eye(3)*a)
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
        return self.lattice.n_cells * self.base_elements

    @property
    def slice_elements(self):
        """int: number of energy elements (orbitals) in a slice of the system"""
        return self.lattice.slice_cells * self.base_elements

    @property
    def n_slices(self):
        """int: Number of slices"""
        return int(self.n_elements / self.slice_elements)

    @property
    def channels(self):
        """int: number of transmission channels of the system"""
        return int(np.prod(self.lattice.shape))

    @property
    def atoms(self):
        """list: List of atom names"""
        return self.lattice.atoms

    def copy(self):
        """TightBinding: Make a deep copy of the model"""
        latt = self.lattice.copy()
        tb = TightBinding(lattice=latt)
        tb.states = self.basis_states
        tb.energies = self.energies
        tb.hopping = self.hopping
        return tb

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

    # =========================================================================
    # Setup
    # =========================================================================

    def add_atom(self, name="A", pos=None, energy=0., orbitals=None):
        """ Add site to lattice cell and store it's energy.

        Parameters
        ----------
        name: str, optional
            Name of the site. The defualt is the origin of the cell
        pos: array_like, optional
            position of site in the lattice cell, default: "A"
        energy: scalar or array_like, optional
            on-site energy of the atom, default: 0.
        orbitals: array_like of str, default: None
            Orbitals

        Raises
        ------
        ValueError:
            raised if position is allready occupied or
            number of orbitals doesn*t match others.
        """
        # Add site to lattice
        self.lattice.add_atom(name, pos)
        states, energy = get_atom_basis(energy, orbitals, self.spin)

        self.energies.append(energy)
        self.basis_states.append(states)
        self.base_elements = sum([len(b) for b in self.basis_states])

    def add_s_atom(self, name="A", pos=None, energy=0.):
        orbitals = "s"
        self.add_atom(name, pos, energy, orbitals)

    def add_p3_atom(self, name="A", pos=None, energy=0.):
        orbitals = ["p_x", "p_y", "p_z"]
        self.add_atom(name, pos, energy, orbitals)

    def add_sp3_atom(self, name="A", pos=None, energy=0.):
        orbitals = ["s", "p_x", "p_y", "p_z"]
        self.add_atom(name, pos, energy, orbitals)

    def sort_states(self, mode="spin"):
        for i in range(self.lattice.n_base):
            if mode == "spin":
                states = self.basis_states[i]
                states.sort(key=lambda s: s.spin)
                self.basis_states[i] = states[::-1]
            elif mode == "orb":
                self.basis_states[i].sort(key=lambda s: s.orb)
            else:
                raise ValueError(f"ERROR: Sort-mode {mode} not recognized (Allowed modes: spin, orb)")

    def set_energy(alpha, energy=0., orbitals=None):
        states, energy = get_atom_basis(energy, orbitals, self.spin)
        self.energies[i] = energy
        self.basis_states[i] = states
        self.base_elements = sum([len(b) for b in self.basis_states])

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

    def set_soc(self, coupling):
        if coupling == 0:
            return
        for alpha in range(self.lattice.n_base):
            states = self.basis_states[alpha]
            n = len(states)
            ham_soc = np.zeros((n, n), dtype="complex")
            if self.spin is False:
                raise ValueError("SOC requires two different spin-types")
            for i in range(n):
                for j in range(n):
                    ham_soc[i, j] = get_soc(states[i], states[j])
            self.energies[alpha] = self.energies[alpha] + (ham_soc * coupling)

    def _find_states(self, atom, txt):
        states = self.basis_states[atom]
        n = len(states)
        return [i for i in range(n) if str(states[i]).startswith(txt)]

    def _init_hoppingmatrix(self):
        n = len(self.energies)
        hopping = List2D.empty(n)
        for i in range(n):
            for j in range(n):
                n = self.energies[i].shape[0]
                m = self.energies[j].shape[0]
                hopping[i, j] = np.zeros((n, m), "complex")
        return hopping

    def _set_orbit_hopvalue(self, value, atom1, atom2, orb1, orb2, distidx=0):
        try:
            indices_i = self._find_states(atom1, orb1) if not isinstance(orb1, int) else [orb1]
            indices_j = self._find_states(atom2, orb2) if not isinstance(orb2, int) else [orb2]
        except ValueError:
            return
        hop_arr = self.hopping[distidx][atom1, atom2]
        if hop_arr is not None:
            for i, j in zip(indices_i, indices_j):
                i, j = sorted([i, j])
                try:
                    hop_arr[i, j] = value
                except IndexError:
                    pass
                if orb1 != orb2:
                    try:
                        hop_arr[j, i] = np.conj(value)
                    except IndexError:
                        pass

    def set_hopping(self, t, orb1=None, orb2=None, distidx=0):
        """ Set hopping energies of the model
        """
        if len(self.hopping) <= distidx:
            while len(self.hopping) <= distidx:
                self.hopping.append(self._init_hoppingmatrix())
            self.lattice.calculate_distances(distidx + 1)

        if orb1 is None and orb2 is None:
            orb1 = orb2 = 0
        if orb1 is not None and orb2 is None:
            orb2 = orb1
        for i in range(self.n_base):
            for j in range(self.n_base):
                self._set_orbit_hopvalue(t, i, j, orb1, orb2, distidx)

    def get_hopping(self, atom1, atom2, distidx=0):
        """ Get the saved hopping value for the given distance

        Parameters
        ----------
        atom1: int
            Index of the first atom
        atom2: int
            Index of the second atom
        distidx: int, default: 0
            distance-index

        Returns
        -------
        t: np.ndarray
        """
        # idx = self.lattice.distances.index(dist)
        return self.hopping[distidx][atom1, atom2]

    def get_distance_hopping(self, atom1, atom2, dist):
        """ Get the saved hopping value for the given distance

        Parameters
        ----------
        atom1: int
            Index of the first atom
        atom2: int
            Index of the second atom
        dist: float
            distance-value

        Returns
        -------
        t: np.ndarray
        """
        # idx = self.lattice.distances.index(dist)
        distidx = self.lattice.distances.index(dist)
        return self.hopping[distidx][atom1, atom2]

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
            size = self.n_elements
            self._disorder_onsite = uniform_eye(self.w_eps, size)

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

    # =========================================================================
    # Hamiltonians
    # =========================================================================

    def _ham_data(self, n):
        datalist = List2D.empty(n)
        for i in range(n):
            # Site energies
            a1 = self.lattice.get_alpha(i)
            datalist[i, i] = self.energies[a1]
            # Hopping energies
            for distidx, j in self.lattice.iter_neighbor_indices(i):
                if j > i:
                    a2 = self.lattice.get_alpha(j)
                    try:
                        datalist[i, j] = self.get_hopping(a1, a2, distidx)
                        datalist[j, i] = self.get_hopping(a2, a1, distidx)
                    except IndexError:
                        pass
        return datalist.arr

    def cell_hamiltonian(self):
        if self.n_base == 1:
            return Hamiltonian(self.energies[0])
        else:
            arr = self._ham_data(self.n_base)
            return Hamiltonian.block(arr)

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
            array = List2D.empty(n)
            for i in range(n):
                # Site energies
                a1 = self.lattice.get_alpha(i)
                array[i, i] = self.energies[a1]
                # Hopping energies
                for distidx, j in self.lattice.iter_neighbor_indices(i):
                    if (i < j) and (j < n):
                        a2 = self.lattice.get_alpha(j)
                        array[i, j] = self.get_hopping(a1, a2, distidx)
                        array[j, i] = self.get_hopping(a2, a1, distidx)
            ham = TbHamiltonian.block(array.arr)
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
        """ Iterates through all slice-Hamiltonians of the system

        Yields
        ------
        ham_slice: TbHamiltonian

        """
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
            array = List2D.empty(n)
            for i in range(n):
                a1 = self.lattice.get_alpha(i)
                # Hopping energies
                for dist in range(len(self.lattice.distances)):
                    for j in range(n):
                        a2 = self.lattice.get_alpha(j)
                        if j + n in self.lattice.neighbours[i][dist]:
                            t = self.get_hopping(a1, a2, dist)
                            array[i, j] = t

            ham = TbHamiltonian.block(array.arr)
            self._slice_hop_cache.load(ham)

        return self._slice_hop_cache.read()

    def hamiltonian(self, w_eps=0.):
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
        if not self._ham_cache:
            # Initialize Hamiltonian
            # ----------------------------
            n = self.lattice.n
            array = List2D.empty(n)
            for i in range(n):
                # Site energies
                a1 = self.lattice.get_alpha(i)
                array[i, i] = self.energies[a1]
                # Hopping energies
                for distidx, j in self.lattice.iter_neighbor_indices(i):
                    if j > i:
                        a2 = self.lattice.get_alpha(j)
                        array[i, j] = self.get_hopping(a1, a2, distidx)
                        array[j, i] = self.get_hopping(a2, a1, distidx)
            ham = TbHamiltonian.block(array.arr)
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

    def mean_ipr(self, omega=eta, n_avrg=100):
        ipr = np.zeros(n_avrg)
        for i in range(n_avrg):
            self.shuffle()
            ipr[i] = self.inverse_participation_ratio(omega)
        return np.mean(ipr)

    def transform_cell_hamiltonian(self, k):
        """ Transform the local hamiltonian in to the k-space

        Parameters
        ----------
        k: float
            wave-vector

        Returns
        -------
        ham: Hamiltonain
        """
        n_dist = self.lattice.n_dist
        ham = self.cell_hamiltonian()
        # One atom in the unit cell
        if self.n_base == 1:
            for dist in range(n_dist):
                nn_vecs = self.lattice.neighbour_vectors(alpha=0, dist_idx=dist)
                t = self.get_hopping(atom1=0, atom2=0, distidx=dist)
                ham = ham + t * np.sum([np.exp(1j * np.dot(k, v)) for v in nn_vecs])
                
        # Multiple atoms in the unit cell
        else:
            block_sizes = [x.shape[0] for x in self.energies]
            slices = blockmatrix_slices(ham.shape, block_sizes)
            for i in range(self.n_base):
                r1 = self.lattice.atom_positions[i]
                for j in range(self.n_base):
                    if i != j:
                        for dist in range(n_dist):
                            nn_vecs = self.lattice.neighbour_vectors(alpha=j, dist_idx=dist)
                            idx = slices[i][j]
                            ham[idx] = ham[idx] * sum([np.exp(1j * np.dot(k, v)) for v in nn_vecs])
        return ham

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
        ham = self.transform_cell_hamiltonian(k)
        return ham.eigvals()

    def bands(self, points, n_e=1000, thresh=1e-9, scale=True, verbose=True, cycle=True):
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
        cycle: bool, default: True
            Connect the first and last point

        Returns
        -------
        band_sections: list
        """
        pairs = list(chain(points, cycle=cycle))
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
                e_vals = np.zeros((n_points, self.base_elements))
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

    def plot_bands(self, points, pointnames, n_e=1000, thresh=1e-9, scale=True, verbose=True,
                   show=True, cycle=True):
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
        cycle: bool, default: True
            Connect the first and last point+

        Returns
        -------
        plot: Plot
        """
        bands = self.bands(points, n_e, thresh, scale, verbose, cycle)
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
        """ Plot the lattice of the device

        See Also
        --------
        Lattice.build()
        """
        return self.lattice.show(show, *args, **kwargs)

    def __str__(self):
        string = "TighBinding-model:\n"
        string += "BASE"
        for i in range(self.n_base):
            string += f"\nAtom {i+1} (States: {self.basis_states[i]})"
            string += "\n" + str(self.energies[i])
        return string
