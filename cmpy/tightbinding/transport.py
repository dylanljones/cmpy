# -*- coding: utf-8 -*-
"""
Created on  06 2019
author: dylan

project: cmpy2
version: 1.0
"""
import numpy as np
import scipy.linalg as la
from cmpy.tightbinding.tightbinding import TightBinding
# from cmpy.core.tightbinding import TightBinding
from cmpy.core import greens
from sciutils import Plot, eta, Cache, normalize
from sciutils.terminal import Progress, prange, Symbols


def gamma(sigma):
    """ Calculate the broadening matrix of the lead on side s

    Parameters
    ----------
    sigma: array_like
        self-energy of the lead

    Returns
    -------
    gamma: array_like
    """
    return 1j * (sigma - np.conj(sigma).T)

# =========================================================================
# TIGHT-BINDING LEAD
# =========================================================================


class Lead:
    """ Semi-infinite Lead object """

    def __init__(self, ham_slice, hop_interslice, dec_thresh=1e-10):
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
        omega: float
            energy of the system

        Returns
        -------
        sigma_l: np.ndarray
            self energy of left surface
        sigma_r: np.ndarray
            self energy of right surface
        """
        self.calculate(omega)
        hop = self.hop
        hop_h = np.conj(self.hop).T
        sigma_l = hop_h @ self.gf_l @ hop
        sigma_r = hop_h @ self.gf_r @ hop
        return sigma_l, sigma_r

    def dos(self, omegas, mode="s", verbose=True):
        """ calculate the density of states of the lead in the specified region

        Parameters
        ----------
        omegas: array_like
            energy values for the density of states
        mode: str, optional
            region specification: "s" for surface and "b" for bulk
            the default is the surface dos
        verbose: bool, optional
            if True, print progress, default: True
        Returns
        -------
        dos: np.ndarray
        """
        omegas = np.asarray(omegas)
        n = omegas.shape[0]
        dos = np.zeros(n, dtype="float")
        name = "surface" if mode == "s" else "bulk"
        for i in prange(n, header=f"Calculating {name}-dos", enabled=verbose):
            self.calculate(omegas[i])
            if mode == "s":
                g = self.gf_l
            elif mode == "b":
                g = self.gf_b
            else:
                raise ValueError(f"Mode not supported: {mode}")
            dos[i] = np.trace(g.imag)
        return -1/np.pi * dos


# =============================================================================
# TIGHTBINDING-DEVICE
# =============================================================================


class TbDevice(TightBinding):

    TRANS_REC_THRESH = 500

    def __init__(self, vectors=np.eye(2), lattice=None):
        super().__init__(vectors, lattice)
        self.lead = None
        self.w_eps = 0

        self._omega_cache = Cache()
        self._sigma_cache = Cache()
        self._gamma_cache = Cache()

    @classmethod
    def square(cls, shape=(2, 1), eps=0., t=1., basis=None, name="A", a=1., wideband=False):
        """ Square device prefab with one atom at the origin of the unit cell

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
        self = super(TbDevice, cls).square(shape, eps, t, name, a)
        self.load_lead(wideband)
        return self

    @classmethod
    def chain(cls, size=1, eps=0., t=1., basis=None, name="A", a=1., wideband=False):
        self = super(TbDevice, cls).chain(size, eps, t, name, a)
        self.load_lead(wideband)
        return self

    @classmethod
    def hexagonal(cls, shape=(2, 1), eps1=0., eps2=0., t=1., atom1="A", atom2="B", a=1., wideband=False):
        self = super(TbDevice, cls).hexagonal(shape, eps1, eps2, t, atom1, atom2, a)
        self.load_lead(wideband)
        return self

    @classmethod
    def sc(cls, shape=(2, 1, 1), eps=0., t=1., name="A", a=1., wideband=False):
        self = super(TbDevice, cls).sc(shape, eps, t, name, a)
        self.load_lead(wideband)
        return self

    # =========================================================================

    @property
    def wideband(self):
        """ bool: True if wideband-limit is used"""
        return self.lead is None

    def prepare(self, omega=eta):
        """ Calculate the lead self energy and broadening matrix of the device

        Parameters
        ----------
        omega: complex, optional
            Energy of the system, default is zero (plus broadening)

        Returns
        -------
        sigmas: tuple
            Self-energies of the leads
        gammas: tuple
            Broadening matrices of the leads
        """
        if not self._sigma_cache or self._omega_cache != omega:
            # print("Calculating sigmas")
            if self.wideband:
                sig = 1j * np.eye(self.slice_elements)
                sigmas = [sig, sig]
            else:
                sigmas = self.lead.sigmas(omega)
            gammas = gamma(sigmas[0]), gamma(sigmas[1])
            self._omega_cache.load(omega)
            self._sigma_cache.load(sigmas)
            self._gamma_cache.load(gammas)

        return self._sigma_cache.read(), self._gamma_cache.read()

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
        self.prepare()

    def copy(self):
        """ Make a deep copy of the model

        Returns
        -------
        dev: TbDevice
        """
        latt = self.lattice.copy()
        dev = TbDevice(lattice=latt)
        dev.energies = self.energies
        dev.lead = self.lead
        dev.w_eps = self.w_eps
        return dev

    def clear_cache(self):
        """Clear all cached objects"""
        super().clear_cache()
        self._omega_cache.clear()
        self._sigma_cache.clear()
        self._gamma_cache.clear()

    # def build(self, shape, wideband=False):
    #     super().build(shape)
    #     self.load_lead(wideband)

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
            self._omega_cache.clear()
            self._sigma_cache.clear()
            self._gamma_cache.clear()

    # =========================================================================
    # Hamiltonians
    # =========================================================================

    def get_slice_hamiltonian(self, i):
        """ Get the i-th slice hamiltonian of the model and add disorder and self energies

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
        ham = super().get_slice_hamiltonian(i)

        if i == 0:
            sigma = self._sigma_cache.read()[0]
            return ham + sigma
        elif i == self.lattice.shape[0] - 1:
            sigma = self._sigma_cache.read()[1]
            return ham + sigma
        else:
            return ham

    def slice_iterator(self):
        for i in range(self.n_slices):
            yield self.get_slice_hamiltonian(i)

    def hamiltonian_device(self, w_eps=0.):
        """ Get the full Hamiltonian of the central device (without self energy of the leads)

        See Also
        --------
        TightBinding.hamiltonian

        Parameters
        ----------
        w_eps: float, optional
            on-site disorder strength

        Returns
        -------
        ham: Hamiltonian
        """
        return super().hamiltonian(w_eps)

    def hamiltonian(self, w_eps=0.):
        """ Override TightBinding.hamiltonian to return the effective Hamiltonian

        Overriding the hamiltonian-function ensures that all methods of the TightBinding base-class
        use the effective Hamiltonian rather than the unmodified device Hamiltonain

        See Also
        --------
        TightBinding.hamiltonian

        Parameters
        ----------
        w_eps: float, optional
            on-site disorder strength

        Returns
        -------
        ham: TbHamiltonian
        """
        sigmas = self._sigma_cache.read()
        ham = self.hamiltonian_device(w_eps)
        n = self.n_elements
        # Add self energy at the corners (interface) of the hamiltonian
        ham.add(0, 0, sigmas[0])
        i = n - self.slice_elements
        ham.add(i, i, sigmas[1])
        return ham

    def hamiltonian_eff(self, omega=eta, w_eps=0.):
        """ Get the full effective Hamiltonian of the device setup and set energy if specified

        Notes
        -----
        Mainly here for legacy support

        Parameters
        ----------
        omega: complex, default: 0
            Energy of the system, default is e=0 (plus broadening)
        w_eps: float, optional
            on-site disorder strength

        Returns
        -------
        ham: TbHamiltonian
        """
        if not self._sigma_cache:
            self.prepare(omega)
        return self.hamiltonian(w_eps)

    # =========================================================================
    # Properties
    # =========================================================================

    def inverse_participation_ratio(self, omega=eta):
        ldos = self.ldos(omega)
        return np.sum(np.power(ldos, 2)) / (np.sum(ldos) ** 2)

    def bulk_dos(self, omegas):
        return self.lead.dos(omegas, mode="b")

    def surface_dos(self, omegas):
        return self.lead.dos(omegas, mode="s")

    def ldos_device(self, omegas, banded=False):
        ham = self.hamiltonian_device()
        return ham.ldos(omegas, banded)

    def dos_device(self, omegas, banded=False):
        return np.sum(self.ldos_device(omegas, banded), axis=1)

    def occupation_device(self, omega, banded=False):
        return normalize(self.ldos_device(omega, banded))

    def occupation(self, omega=None, banded=False):
        if omega:
            self.prepare(omega)
        else:
            omega = self._omega_cache.read()
        return super().occupation(omega, banded)

    def transmission(self, omega=eta, rec_thresh=TRANS_REC_THRESH):
        """ Calculate the transmission of the tight binding device

        If the size of the hamiltonian is smaller then the given threshold
        the full hamiltonian is used, otherwise the rgf-algorithm is used.
        For the rgf the hamiltonian is built up blockwise.

        Parameters
        ----------
        omega: float, optional
            Energy of the system, default is zero (plus broadening)
        rec_thresh: int, default: TRANS_REC_THRESH
            If size of Hamiltonian matrix is larger than the threshold the recursive algorithm will be used

        Returns
        -------
        t: float
        """
        # Calculate self energy and broadening matrix of the leads
        _, gammas = self.prepare(omega)

        blocksize = self.slice_elements
        # Use RGF-algorithm
        # =================
        if self.n_elements >= rec_thresh:
            # Calculate lower left corner of greens function (rgf)
            # ----------------------------------------------------
            e = np.eye(blocksize) * omega
            hop = self.slice_hopping()
            hop_adj = np.conj(hop).T

            # Calculate gf block using left interface of the hamiltonain with self energy added
            h_eff = self.get_slice_hamiltonian(0)
            g_nn = la.inv(e - h_eff, overwrite_a=True, check_finite=False)
            g_1n = g_nn
            # Calculate gf blocks using bulk and right interface blocks of the hamiltonain
            for i in range(1, self.n_slices):
                h_eff = self.get_slice_hamiltonian(i) + hop @ g_nn @ hop_adj
                g_nn = la.inv(e - h_eff, overwrite_a=True, check_finite=False)
                g_1n = g_1n @ hop_adj @ g_nn

        # Use full Hamiltonian
        # ====================
        else:
            ham = self.hamiltonian()
            # calculate greens-function by inverting full hamiltonian
            g = greens.gf(ham, omega)
            # Get lower left block of gf
            g_1n = g[-blocksize:, :blocksize]

        return np.trace(gammas[1] @ g_1n @ gammas[0] @ g_1n.conj().T).real

    def transmission_curve(self, omegas=None, verbose=True):
        """ Calculate the transmission curve for multiple energy values

        Parameters
        ----------
        omegas: array_like, default: NOne
            energy-values of the transmission curve. If none given the
            standard range from -5 to 5 is used (n=100)
        verbose: bool, default: True
            print progress if True

        Returns
        -------
        trans: np.ndarray
        """
        self.shuffle()
        if omegas is None:
            n = 100
            omegas = np.linspace(-5, 5, 100) + eta
        else:
            n = omegas.shape[0]
        trans = np.zeros(n, dtype="float")
        for i in prange(n, header="Calculating transmission", enabled=verbose):
            trans[i] = self.transmission(omegas[i])
        return omegas, trans

    def transmission_mean(self, omega=eta, n=100, flatten=True, prog=None, header=None):
        """ Calculate the mean transmission of the device

        Parameters
        ----------
        omega: float, optional
            Energy of the system, default is zero (plus broadening)
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
        trans = np.zeros(n)
        for i in range(n):
            p.update()
            trans[i] = self.transmission(omega)
        if prog is None:
            p.end()
        return np.mean(np.log(trans)) if flatten else trans

    def transmission_loss(self, lengths, omega=eta, n_avrg=1000, flatten=True, prog=None, header=None):
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

        for i in range(n):
            length = lengths[i]
            self.reshape(length)
            p.set_description(f"Length: {length}")
            for j in range(n_avrg):
                p.update()
                self.shuffle()
                trans[i, j] = self.transmission(omega)
        if prog is None:
            p.end()
        return np.mean(np.log(trans), axis=1) if flatten else trans

    def normal_transmission(self, omega=eta):
        """ calculate normal transmission of the device without any disorder

        Parameters
        ----------
        omega: float, optional
            Energy of the system, default is zero (plus broadening)

        Returns
        -------
        t: float
        """
        # Store device configuration
        length = self.lattice.shape[0]
        w = self.w_eps
        # Set shortest model with no disorder
        self.reshape(5)
        self.set_disorder(0)
        # Calculate transmission
        t = self.transmission(omega)
        # Reset device configuration
        self.reshape(length)
        self.set_disorder(w)

        return t

    # =========================================================================

    def plot_transmission_hist(self, omega=eta, n=1000, show=True):
        """ Plot the histogram of the transmission for the disordered system

        Parameters
        ----------
        omega: float, optional
            Energy of the system, default is zero (plus broadening)
        n: int, optional
            Number of sample points, default: 1000
        show: bool, optional
            If True, show plot. The default is True
        """
        if self.w_eps == 0:
            raise ValueError("Histogram can't be computed: No Disorder set")

        trans = self.transmission_mean(omega=omega, n=n, flatten=False)
        mean = np.mean(trans)
        std = np.std(trans)
        print(f"<T>={mean:.5}, {Symbols.Delta}T={std:.5}")

        plot = Plot()
        plot.set_scales(xscale="log")

        bins = np.geomspace(min(trans), 1, 100)
        plot.histogram(trans, bins=bins)
        if show:
            plot.show()
        return plot
