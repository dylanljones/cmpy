# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2022, Dylan Jones

import itertools
import numpy as np
from scipy import sparse
from scipy import linalg as la
from abc import abstractmethod
from lattpy import Lattice
from .abc import AbstractModel


def eigvalsh_chain(num_sites, eps, t):
    """Computes the eigenvalues of the Hamiltonain of a 1D tight-binding model.

    Parameters
    ----------
    num_sites : int
        The number of lattice sites N in the model.
    eps : float or (N) np.ndarray
        The on-site energy of the model
    t : float
        The hopping energy of the model.

    Returns
    -------
    eigvals : (N) np.ndarray
        The eigenvalues of the Hamiltonian.
    eigvecs : (N, N) np.ndarray
        The eigenvectors of the Hamiltonian.
    """
    if isinstance(eps, (float, int, complex)):
        diag = eps * np.ones(num_sites)
    else:
        diag = eps
    off_diag = t * np.ones(num_sites - 1)
    return la.eigvalsh_tridiagonal(diag, off_diag)


def eigh_chain(num_sites, eps, t):
    """Computes the eigen-values and -vectors of the Hamiltonain of a 1D model.

    Parameters
    ----------
    num_sites : int
        The number of lattice sites N in the model.
    eps : float or (N) np.ndarray
        The on-site energy of the model
    t : float
        The hopping energy of the model.

    Returns
    -------
    eigvals : (N) np.ndarray
        The eigenvalues of the Hamiltonian.
    eigvecs : (N, N) np.ndarray
        The eigenvectors of the Hamiltonian.
    """
    if isinstance(eps, (float, int, complex)):
        diag = eps * np.ones(num_sites)
    else:
        diag = eps
    off_diag = t * np.ones(num_sites - 1)
    return la.eigh_tridiagonal(diag, off_diag)


class AbstractTightBinding(Lattice, AbstractModel):
    """Abstract Tight-binding model based on a lattice.

    Parameters
    ----------
    vectors : (N, N) array_like
        The basis vectors of a lattice.
    """

    def __init__(self, vectors):
        AbstractModel.__init__(self)
        Lattice.__init__(self, vectors)
        self.path = None

    @abstractmethod
    def get_energy(self, alpha=0):
        """Returns the on-site energy of an atom in the unit-cell of the lattice.

        Parameters
        ----------
        alpha : int, optional
            The index of the atom.

        Returns
        -------
        energy: array_like
            The on-site energy of the atom. Can either be a scalar or a square matrix.
        """
        pass

    @abstractmethod
    def get_hopping(self, distidx=0):
        """Returns the hopping-site energy between atoms with a certain distance.

        Parameters
        ----------
        distidx : int, optional
            The distance index of the atom-pair.

        Returns
        -------
        energy: array_like
            The hopping energy. Can either be a scalar or a square matrix.
        """
        pass

    def analyze(self) -> None:
        super().analyze()
        self.finalize()

    def finalize(self):
        """Called after analyzing the lattice. Parameters should be initialized here."""
        pass

    def hamiltonian_cell(self, dtype=None):
        """Constructs the hamiltonian of the unit-cell.

        Parameters
        ----------
        dtype : str or np.dtype or type, optional
            Optional datatype of the resulting matrix.

        Returns
        -------
        ham : (N, N) np.ndarray
            The hamiltonian matrix of the unit-cell. The shape is the number of atoms
            in the unit-cell.
        """
        ham = np.zeros((self.num_base, self.num_base), dtype=dtype)
        for alpha in range(self.num_base):
            ham[alpha, alpha] = self.get_energy(alpha)
            for distidx in range(self.num_distances):
                t = self.get_hopping(distidx)
                for idx in self.get_neighbors(alpha=alpha, distidx=distidx):
                    alpha2 = idx[-1]
                    ham[alpha, alpha2] = t
        return ham

    def hamiltonian_data(self, dtype=None):
        """Computes the elements of the hamiltonian.

        Parameters
        ----------
        dtype : str or np.dtype or type, optional
            Optional datatype of the data.

        Returns
        -------
        rows : (N, ) np.ndarray
            The row indices of the elements.
        cols : (N, ) np.ndarray
            The column indices of the elements.
        data : (N, ) np.ndarray
            The elements of the hamiltonian matrix.
        """
        dmap = self.data.map()
        data = np.zeros(dmap.size, dtype=dtype)
        for alpha in range(self.num_base):
            data[dmap.onsite(alpha)] = self.get_energy(alpha)
        for distidx in range(self.num_distances):
            data[dmap.hopping(distidx)] = self.get_hopping(distidx)
        rows, cols = dmap.indices
        return rows, cols, data

    def hamiltonian(self, dtype=None):
        """Constructs the hamiltonian as a sparse matrix in CSR format.

        Parameters
        ----------
        dtype : str or np.dtype or type, optional
            Optional datatype of the resulting matrix.

        Returns
        -------
        ham : (N, N) sparse.csr_matrix
            The hamiltonian matrix in sparse format.
        """
        rows, cols, data = self.hamiltonian_data(dtype)
        arg = data, (rows, cols)
        return sparse.csr_matrix(arg)

    def get_neighbor_vectors_to(self, alpha1, alpha2, distidx=0):
        """Computes the neighbor vector between two sites."""
        keys = list(sorted(self._base_neighbors[alpha1].keys()))
        dist = keys[distidx]
        indices = self._base_neighbors[alpha1][dist]
        indices = indices[indices[:, -1] == alpha2]
        pos0 = self._positions[alpha1]
        positions = self.get_positions(indices)
        return positions - pos0

    def hamiltonian_kernel(self, k, ham_cell=None):
        """Computes the fourier transformed hamiltonian of the unit-cell.

        Parameters
        ----------
        k : (N, ) array_like
            The point in frequency-space.
        ham_cell : (N, N) array_like, optional
            Optional cell-hamiltonian in real-space. If ``None`` the hamiltonian
            will be constructed.

        Returns
        -------
        ham_k : (N, N) np.ndarray
            The transformed hamiltonian.
        """
        if ham_cell is None:
            ham_cell = self.hamiltonian_cell(dtype=np.complex64)
        ham = ham_cell.copy()

        if self.num_base == 1:
            ham = np.array([[self.get_energy(0)]], dtype=np.complex64)
            for distidx in range(self.num_distances):
                ham += self.get_hopping(distidx) * self.fourier_weights(
                    k, distidx=distidx
                )
            return ham

        for alpha in range(self.num_base):
            ham[alpha, alpha] = self.get_energy(alpha)
            for distidx in range(self.num_distances):
                for alpha2 in range(alpha + 1, self.num_base):
                    vecs = self.get_neighbor_vectors_to(alpha, alpha2, distidx)
                    ham[alpha, alpha2] *= np.sum(np.exp(1j * np.inner(k, +vecs)))
                    ham[alpha2, alpha] *= np.sum(np.exp(1j * np.inner(k, -vecs)))

        return ham

    def dispersion(self, k, mu=0.0, sort=False):
        """Computes the energy dispersion for one or multiple points in frequency-space.

        Parameters
        ----------
        k : (..., N) array_like
            The point(s) in frequency-space.
        mu : float, optional
            The chemical potential.
        sort : bool, optional
            Flag if energy values are sorted. The default is ``False``.

        Returns
        -------
        disp : (..., N) np.ndarray
            The energy values for the given point(s).
        """
        k = np.atleast_2d(k)
        disp = np.zeros((len(k), self.num_base), dtype=np.float32)
        ham_cell = self.hamiltonian_cell()
        for i, _k in enumerate(k):
            ham_k = self.hamiltonian_kernel(_k, ham_cell)
            eigvals = la.eigvalsh(ham_k).real
            if sort:
                eigvals = np.sort(eigvals)
            disp[i] = eigvals
        return (disp[0] if len(k) == 1 else disp) - mu

    def bands(self, nums=100, mu=0.0, sort=False, offset=0.0, check=True):
        brillouin = self.brillouin_zone()
        k_ranges = brillouin.linspace(nums, offset)
        lengths = [len(k) for k in k_ranges]
        bands = np.zeros((*lengths, self.num_base))
        ham_cell = self.hamiltonian_cell()
        for item in itertools.product(*[range(n) for n in lengths]):
            k = np.array([k_ranges[i][item[i]] for i in range(len(k_ranges))])
            if not check or brillouin.check(k):
                ham_k = self.hamiltonian_kernel(k, ham_cell)
                eigvals = la.eigvalsh(ham_k).real
                if sort:
                    eigvals = np.sort(eigvals)
                bands[item] = eigvals
            else:
                bands[item] = np.nan
        return k_ranges, bands.T - mu


class BaseTightBindingModel(AbstractTightBinding):
    def __init__(self, vectors):
        super().__init__(vectors)

    def set_energies(self, *eps):
        self.set_param("eps", np.array(eps))

    def set_hopping(self, *t):
        self.set_param("hop", np.array(t))

    def get_energy(self, alpha=0):
        return self.eps[alpha]

    def get_hopping(self, distidx=0):
        return self.hop[distidx]

    def finalize(self):
        self.set_param("eps", np.zeros(self.num_base))
        self.set_param("hop", np.ones(self.num_distances))
