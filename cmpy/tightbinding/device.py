# -*- coding: utf-8 -*-
"""
Created on 27 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
import numpy as np
from cmpy.core import greens, Matrix
from cmpy.core import prange, Progress
from .tightbinding import TightBinding


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


class TbDevice(TightBinding):

    def __init__(self, vectors=np.eye(2)):
        super().__init__(vectors)
        self.lead = None
        self.w_eps = 0

    @classmethod
    def square(cls, shape, eps=0., t=1., name="A", a=1., wideband=False):
        self = cls(a * np.eye(2))
        self.add_atom(name, eps)
        self.set_hopping(t)
        self.build(shape)
        self.load_lead(wideband)
        return self

    @property
    def wideband(self):
        return self.lead is None

    def load_lead(self, wideband=False):
        if wideband:
            self.lead = None
        else:
            ham_slice = self.slice_hamiltonian()
            hop = self.slice_hopping()
            self.lead = Lead(ham_slice, hop)

    def reshape(self, x=None, y=None, z=None):
        self.lattice.reshape(x, y, z)
        self.load_lead(self.wideband)

    def set_disorder(self, w_eps):
        self.w_eps = w_eps

    def prepare(self, omega):
        if self.wideband:
            sig = 1j * np.eye(self.slice_elements)
            sigmas = [sig, sig]
        else:
            sigmas = self.lead.sigmas(omega)
        gammas = gamma(sigmas[0]), gamma(sigmas[1])
        return sigmas, gammas

    # =========================================================================

    def _expand_gammas(self, gammas, size):
        n = self.slice_elements
        if n != size:
            gamma_l = np.zeros((size, size), "complex")
            gamma_l[:n, :n] = gammas[0]
            gamma_r = np.zeros((size, size), "complex")
            gamma_r[size - n:, size - n:] = gammas[1]
            gammas = gamma_l, gamma_r
        return gammas

    def transmission(self, omega, sigmas=None, gammas=None):
        n = self.n_elements
        n_s = self.slice_elements
        ham = Matrix(self.hamiltonian(self.w_eps))

        if sigmas is None:
            sigmas, gammas = self.prepare(omega)

        # Add sigmas at corners of hamiltonian
        ham[:n_s, :n_s] += sigmas[0]
        ham[n - n_s:, n - n_s:] += sigmas[1]

        chunksize = 1 * n_s
        if chunksize != n_s:
            gammas = self._expand_gammas(gammas, chunksize)
        g_1n = greens.rgf(ham, omega, chunksize)
        return np.trace(gammas[1] @ g_1n @ gammas[0] @ g_1n.conj().T).real

    def mean_transmission(self, omega, sigmas=None, gammas=None, n=100, prog=None):
        if sigmas is None:
            sigmas, gammas = self.prepare(omega)
        trans = np.zeros(n)
        for i in range(n):
            if prog is not None:
                prog.update()
            trans[i] = self.transmission(omega, sigmas, gammas)
        return np.mean(trans)

    def transmission_curve(self, omegas, verbose=True):
        n = omegas.shape[0]
        trans = np.zeros(n, dtype="float")
        for i in prange(n, header="Calculating transmission", enabled=verbose):
            trans[i] = self.transmission(omegas[i])
        return trans

    def transmission_loss(self, omega, lengths, n_avrg=100, header=None, flatten=False):
        n = lengths.shape[0]
        trans = np.zeros((n, n_avrg))
        sigmas, gammas = self.prepare(omega)
        with Progress(total=n * n_avrg, header=header) as p:
            for i in range(n):
                length = lengths[i]
                p.set_description(f"Length: {length}")
                self.reshape(length)
                for j in range(n_avrg):
                    p.update()
                    trans[i, j] = self.transmission(omega, sigmas, gammas)
        if flatten:
            trans = np.mean(trans, axis=1)
        return trans


class Lead:

    def __init__(self, ham_slice, hop_interslice, dec_thresh=1e-100):
        self.omega = 0
        self.ham = ham_slice
        self.hop = hop_interslice

        self._thresh = dec_thresh
        n = ham_slice.shape[0]
        self._gfs = np.zeros((3, n, n), dtype="complex")

    @property
    def gf_l(self):
        return self._gfs[0]

    @property
    def gf_b(self):
        return self._gfs[1]

    @property
    def gf_r(self):
        return self._gfs[2]

    def calculate(self, omega):
        if np.imag(omega) == 0:
            raise ValueError("Omega must have imagninary part")
        if omega != self.omega:
            self.omega = omega
            self._gfs = greens.rda(self.ham, self.hop, omega, self._thresh)

    def sigmas(self, omega):
        self.calculate(omega)
        t_lc = self.hop
        t_cl = self.hop.conj().T
        t_rc = t_lc
        t_cr = t_cl
        sigma_l = t_cl @ self.gf_l @ t_lc
        sigma_r = t_cr @ self.gf_r @ t_rc
        return sigma_l, sigma_r

    def dos(self, omegas, mode="s", verbose=True):
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


def square_device(eps=0., t=1., shape=(2, 1), name="A", a=1., wideband=False):
    """ square device prefab with one atom at the origin of the unit cell

    Parameters
    ----------
    eps: float, optional
        energy of the atom, the default is 0
    t: float, otional
        hopping parameter, the default is 1.
    shape: tuple, optional
        shape to build lattice, the default is (1, 1)
        if None, the lattice won't be built on initialization
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
    device = TbDevice(a * np.eye(2))
    device.add_atom(name=name, energy=eps)
    device.set_hopping(t)
    device.build(shape)
    device.load_lead(wideband)
    return device
