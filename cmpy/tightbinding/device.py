# -*- coding: utf-8 -*-
"""
Created on 27 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
import numpy as np
from scipy import linalg as la
from cmpy.core import greens, prange, Progress, eta
from .basis import sp3_basis, p3_basis
from .tightbinding import TightBinding

# =============================================================================
# TIGHTBINDING-LEAD
# =============================================================================


class Lead:
    """ Semi-infinite Lead object """

    def __init__(self, ham_slice, hop_interslice, dec_thresh=1e-100):
        self.omega = 0
        self.ham = ham_slice
        self.hop = hop_interslice

        self._thresh = dec_thresh
        n = ham_slice.shape[0]
        self._gfs = np.zeros((3, n, n), dtype="complex")

    @property
    def gf_l(self):
        """ np.ndarray: left surface greens function of the semi-infinite lead"""
        return self._gfs[0]

    @property
    def gf_b(self):
        """ np.ndarray: bulk greens function of the semi-infinite lead"""
        return self._gfs[1]

    @property
    def gf_r(self):
        """ np.ndarray: right surface greens function of the semi-infinite lead"""
        return self._gfs[2]

    def calculate(self, omega):
        """ calculate surface and bulk green functions of semiinfinite lead

        Parameters
        ----------
        omega: float
            energy of the system
        """
        if np.imag(omega) == 0:
            raise ValueError("Omega must have imagninary part")
        if omega != self.omega:
            self.omega = omega
            self._gfs = greens.rda(self.ham, self.hop, omega, self._thresh)

    def sigmas(self, omega):
        """ calculate self energies of the semiinfinite lead

        Parameters
        ----------
        omega: floatDEVICE
            energy of the system

        Returns
        -------
        sigma_l: np.ndarray
            self energy of left surface
        sigma_r: np.ndarray
            self energy of right surface
        """
        self.calculate(omega)
        t_lc = self.hop
        t_cl = self.hop.conj().T
        t_rc = t_lc
        t_cr = t_cl
        sigma_l = t_cl @ self.gf_l @ t_lc
        sigma_r = t_cr @ self.gf_r @ t_rc
        return sigma_l, sigma_r

    def dos(self, omegas, mode="s", verbose=True):
        """ calculate the density of states of the lead in the specified region

        Parameters
        ----------
        omegas: array_like
            energy values for the density of states
        mode: str, optional
            region specification: "s" for surface and "b" for bulk
            the defauolt is the surface dos
        verbose: bool, optional
            if True, print progress, default: True
        Returns
        -------
        dos: np.ndarray
        """
        omegas = np.asarray(omegas)
        n = omegas.shape[0]
        dos = np.zeros(n)
        name = "surface" if mode == "s" else "center"
        for i in prange(n, header=f"Calculating {name}-dos", enabled=verbose):
            self.calculate(omegas[i])
            if mode == "s":
                g = self.gf_l
            elif mode == "c":
                g = self.gf_b
            else:
                raise ValueError(f"Mode not supported: {mode}")
            dos[i] = np.trace(g.imag)
        return -1/np.pi * dos


def gamma(sigma_s):
    """ Calculate the broadening matrix of the lead on side s

    Parameters
    ----------
    sigma_s: array_like
        self-energy of the lead

    Returns
    -------
    gamma: array_like
    """
    return 1j * (sigma_s - np.conj(sigma_s).T)


# =============================================================================
# TIGHTBINDING-DEVICE
# =============================================================================


class TbDevice(TightBinding):

    def __init__(self, vectors=np.eye(2), lattice=None):
        super().__init__(vectors, lattice)
        self.lead = None
        self.w_eps = 0

        self._cached_omega = None
        self._cached_sigmas = None
        self._cached_gammas = None

    @classmethod
    def square(cls, shape=(2, 1), eps=0., t=1., basis=None, name="A", a=1., wideband=False):
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
        wideband: bool, optional
            if True, use wide band approximation, the default is False

        Returns
        -------
        latt: Lattice
        """
        self = cls(a * np.eye(2))
        if basis is not None:
            eps = basis.eps
            t = basis.hop
        self.add_atom(name, energy=eps)
        self.set_hopping(t)
        if shape is not None:
            self.build(shape)
            self.load_lead(wideband)
        return self

    @classmethod
    def square_sp3(cls, shape=(2, 1), basis=sp3_basis(), name="A", a=1., wideband=False):
        """ square device prefab with one atom at the origin of the unit cell with p orbitals

        Parameters
        ----------
        shape: tuple, optional
            shape to build lattice, the default is (1, 1)
            if None, the lattice won't be built on initialization
        basis: Basis, optional
            energy basis of one atom, if None, use default sp3-configuration
        name: str, optional
            name of the atom, the default is "A"
        a: float, optional
            lattice constant, default: 1
        wideband: bool, optional
            if True, use wide band approximation, the default is False

        Returns
        -------
        latt: Lattice
        """
        return cls.square(shape, basis=basis, name=name, a=a, wideband=wideband)

    @classmethod
    def square_p3(cls, shape=(2, 1), basis=p3_basis(), name="A", a=1., wideband=False):
        """ square device prefab with one atom at the origin of the unit cell with s and p orbitals

        Parameters
        ----------
        shape: tuple, optional
            shape to build lattice, the default is (1, 1)
            if None, the lattice won't be built on initialization
        basis: Basis, optional
            energy basis of one atom, if None, use default p3-configuration
        name: str, optional
            name of the atom, the default is "A"
        a: float, optional
            lattice constant, default: 1
        wideband: bool, optional
            if True, use wide band approximation, the default is False

        Returns
        -------
        latt: Lattice
        """
        return cls.square(shape, basis=basis, name=name, a=a, wideband=wideband)

    @property
    def wideband(self):
        """ bool: True if wideband-limit is used"""
        return self.lead is None

    def load_lead(self, wideband=False):
        """ Load the leads of the model

        Parameters
        ----------
        wideband: bool, optional
            if True, use wideband-limit, otherwise use semiinfinite
            leads of the same lattice structure. The default is False
        """
        if wideband:
            self.lead = None
        else:
            ham_slice = self.slice_hamiltonian()
            hop = self.slice_hopping()
            self.lead = Lead(ham_slice, hop)

    def copy(self):
        """ Make a deep copy of the model

        Returns
        -------
        dev: TbDevice
        """
        latt = self.lattice.copy()
        dev = TbDevice(lattice=latt)
        dev.n_orbs = self.n_orbs
        dev.energies = self.energies
        dev.hoppings = self.hoppings
        dev.lead = self.lead
        dev.w_eps = self.w_eps
        return dev

    def clear_cache(self):
        """Clear all cached objects"""
        super().clear_cache()
        self._cached_omega = None
        self._cached_sigmas = None
        self._cached_gammas = None

    def reshape(self, x=None, y=None, z=None):
        """ Reshape lattice of the device and reload leads

        Parameters
        ----------
        x: int, optional
            new size in x-direction
        y: int, optional
            new size in y-direction
        z: int, optional
            new size in z-direction
        """
        super().reshape(x, y, z)
        self.load_lead(self.wideband)
        if (y is not None) or (z is not None):
            self._cached_omega = None
            self._cached_sigmas = None
            self._cached_gammas = None

    def set_disorder(self, w_eps):
        """ Set the amoount of disorder of the on-site energies

        Parameters
        ----------
        w_eps: float
            disorder amount
        """
        self.w_eps = w_eps

    def show(self):
        """ Plot the lattice of the device"""
        self.lattice.show()

    # =========================================================================

    def prepare(self, omega=eta):
        if (self._cached_sigmas is None) or (omega != self._cached_omega):
            print("Calculating sigmas")
            if self.wideband:
                sig = 1j * np.eye(self.slice_elements)
                sigmas = [sig, sig]
            else:
                sigmas = self.lead.sigmas(omega)
            gammas = gamma(sigmas[0]), gamma(sigmas[1])
            self._cached_omega = omega
            self._cached_sigmas = sigmas
            self._cached_gammas = gammas

        return self._cached_sigmas, self._cached_gammas

    def transmission(self, omega=eta, sigmas=None, gammas=None, rec_thresh=500, compressed=True):
        """ Calculate the transmission of the tight binding device

        If the size of the full hamiltonian is smaller then the given threshold
        the full hamiltonian is used, otherwise the rgf-algorithm is used. For the rgf
        the hamiltonian is built up blockwise.

        Parameters
        ----------
        omega: float, optional
            Energy of the system, default is zero (plus broadening)
        sigmas: tuple, optional
            Self-energy of the leads, default is None.
            If not given, sigmas will be calculated
        gammas: tuple, optional
            Broadening matrix of the leads, default is None.
            If not given, gammas will be calculated
        rec_thresh: int, optional
            Threshold to use recursive greens function algorithm, default is 500.
        compressed: bool, optional
            if True, build up slice-hamiltonian for each iteration of the RGF-Algorithm.
            default: True.

        Returns
        -------
        t: float
        """
        # Calculate self energy and broadening matrix of the leads
        if sigmas is None:
            sigmas, gammas = self.prepare(omega)

        blocksize = self.slice_elements

        # =============================
        # Use RGF-algorithm
        # =============================
        if self.n_elements > rec_thresh:
            # If compressed, build up slice-hamiltonian for each iteration
            # of the RGF-Algorithm.
            if compressed:
                # Calculate lower left corner of greens function (rgf)
                n_blocks = self.lattice.shape[0]
                h_hop = self.slice_hopping()

                e = np.eye(blocksize) * omega
                # Calculate gf block using left interface of the hamiltonain with self energy added
                g_nn = la.inv(e - (self.slice_hamiltonian(self.w_eps) + sigmas[0]), overwrite_a=True)
                g_1n = g_nn

                # Calculate gf block using bulk blocks of the hamiltonain
                for i in range(1, n_blocks-1):
                    h = self.slice_hamiltonian(self.w_eps) + h_hop @ g_nn @ np.conj(h_hop).T
                    g_nn = la.inv(e - h, overwrite_a=True)
                    g_1n = g_1n @ h_hop @ g_nn

                # Calculate gf block using right interface of the hamiltonain with self energy added
                h = self.slice_hamiltonian(self.w_eps) + sigmas[1] + h_hop @ g_nn @ np.conj(h_hop).T
                g_nn = la.inv(e - h, overwrite_a=True)
                g_1n = g_1n @ h_hop @ g_nn

            # Otherwise, use blocks of full hamiltonian.
            else:
                ham = self.hamiltonian(self.w_eps)
                n = ham.n
                # Add self energy at the corners (interface) of the hamiltonian
                ham.add(0, 0, sigmas[0])
                i = n - self.slice_elements
                ham.add(i, i, sigmas[1])

                # Check if hamiltonian-blocks are configured right
                if ham.block_size is None or ham.block_shape[0] != int(n / blocksize):
                    ham.config_blocks(blocksize)
                # Use recursive greens-function algorithm to calculate lower left block of gf
                g_1n = greens.rgf(ham, omega)

        # =============================
        # Use full Hamiltonian
        # =============================
        else:
            ham = self.hamiltonian(self.w_eps)
            n = self.n_elements
            # Add self energy at the corners (interface) of the hamiltonian
            ham.add(0, 0, sigmas[0])
            i = n - self.slice_elements
            ham.add(i, i, sigmas[1])

            # calculate greens-function by inverting full hamiltonian
            g = greens.greens(ham, omega)
            # Get lower left block of gf
            g_1n = g[-blocksize:, :blocksize]

        return np.trace(gammas[1] @ g_1n @ gammas[0] @ g_1n.conj().T).real

    def transmission_curve(self, omegas, verbose=True):
        """ Calculate the transmission curve for multiple energy values

        Parameters
        ----------
        omegas: array_like
            energy-values

        Returns
        -------
        trans: np.ndarray
        """
        n = omegas.shape[0]
        trans = np.zeros(n, dtype="float")
        for i in prange(n, header="Calculating transmission", enabled=verbose):
            trans[i] = self.transmission(omegas[i])
        return trans

    def mean_transmission(self, omega=eta, sigmas=None, gammas=None, n=100,
                          flatten=True, prog=None, header=None):
        """ Calculate the mean transmission of the device

        Parameters
        ----------
        omega: float, optional
            Energy of the system, default is zero (plus broadening)
        sigmas: tuple, optional
            Self-energy of the leads, default is None.
            If not given, sigmas will be calculated
        gammas: tuple, optional
            Broadening matrix of the leads, default is None.
            If not given, gammas will be calculated
        n: int, optional
            number of values to calculate, the default is 100
        flatten: bool, optional
            if True, return mean of calculated data. Otherwise
            return all values. The default is True
        prog: Progress, optional
            optional outer Progress object
        header: str, optional
            If no outer Progress object is used, set header of local
            Progress object

        Returns
        -------
        trans: sclalar or np.ndarray
        """
        p = Progress(total=n, header=header) if prog is None else prog
        if sigmas is None:
            sigmas, gammas = self.prepare(omega)
        trans = np.zeros(n)
        for i in range(n):
            p.update()
            trans[i] = self.transmission(omega, sigmas, gammas)
        if prog is None:
            p.end()
        return np.mean(trans) if flatten else trans

    def transmission_loss(self, lengths, omega=eta, n_avrg=1000, flatten=False,
                          prog=None, header=None):
        """ Calculate the mean transmission loss of the device
        
        Parameters
        ----------
        lengths: array_like
            lengths of the system to use in calculation
        omega: float, optional
            Energy of the system, default is zero (plus broadening)
        n_avrg: int, otional
            number os samples per length, default is 1000
        flatten: bool, optional
            if True, return mean of calculated data. Otherwise
            return all values. The default is True
        prog: Progress, optional
            optional outer Progress object
        header: str, optional
            If no outer Progress object is used, set header of local
            Progress object

        Returns
        -------
        trans: sclalar or np.ndarray
        """
        n = lengths.shape[0]
        p = Progress(total=n * n_avrg, header=header) if prog is None else prog

        trans = np.zeros((n, n_avrg))
        sigmas, gammas = self.prepare(omega)
        for i in range(n):
            length = lengths[i]
            self.reshape(length)
            p.set_description(f"Length: {length}")
            for j in range(n_avrg):
                p.update()
                trans[i, j] = self.transmission(omega, sigmas, gammas)
        if prog is None:
            p.end()
        return np.mean(trans, axis=1) if flatten else trans
