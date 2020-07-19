# coding: utf-8
"""
Created on 08 Jul 2020
Author: Dylan Jones
"""
import numpy as np


def bethe_dos(z, t):
    """ Density of states of the bethe lattice in infinite dimensions.

    Parameters
    ----------
    z: complex ndarray or complex
        Green's function is evaluated at complex frequency .math:'z'.
    t: float
        Hopping parameter of the lattice model.

    Returns
    -------
    bethe_dos: complex ndarray or complex
    """
    energy = np.asarray(z).clip(-2 * t, 2 * t)
    return np.sqrt(4 * t**2 - energy**2) / (2 * np.pi * t**2)


def bethe_gf_omega(z, half_bandwidth=1):
    """Local Green's function of Bethe lattice for infinite Coordination number.

    References
    ----------
        Taken from gf_tools by Weh Andreas: https://github.com/DerWeh/gftools/blob/master/gftools/__init__.py

    Parameters
    ----------
    z : complex ndarray or complex
        Green's function is evaluated at complex frequency `z`
    half_bandwidth : float
        half-bandwidth of the DOS of the Bethe lattice
        The `half_bandwidth` corresponds to the nearest neighbor hopping `t=D/2`
    Returns
    -------
    bethe_gf_omega : complex ndarray or complex
        Value of the Green's function
    """
    z_rel = z / half_bandwidth
    return 2. / half_bandwidth * z_rel * (1 - np.sqrt(1 - 1 / (z_rel * z_rel)))


def self_energy(gf_imp0, gf_imp):
    """ Calculate the self energy from the non-interacting and interacting Green's function"""
    return 1/gf_imp0 - 1/gf_imp


def mix_values(old: float, new: float, mixing: float = 1.0) -> float:
    """ Mixes the old and new value of a parameter.

    Parameters
    ----------
    old: float
        The old value of the parameter.
    new: float
        The new value of the parameter.
    mixing: float, optional
        The proportion of the new value. The default is `1` (no mixing).

    Returns
    -------
    mixed: float
    """
    return new if mixing >= 1.0 else new * mixing + old * (1.0 - mixing)


class ErrorStats:

    def __init__(self):
        self._errors = list()

    @property
    def iterations(self):
        return len(self._errors)

    @property
    def errors(self):
        return self._errors[-1]

    @property
    def history(self):
        return np.asarray(self._errors).T

    def update(self, *values):
        self._errors.append(list(values))

    def __iter__(self):
        yield from np.atleast_1d(self.errors)

    def __format__(self, format_spec):
        return ", ".join([f"{err:{format_spec}}" for err in self.errors])

    def __repr__(self):
        return f"{self.__class__.__name__}(it: {self.iterations}, errs: " + self.__format__(".1e") + ")"

    def __str__(self):
        lines = list()
        lines.append(f"It: {self.iterations}")
        lines.append(f"Δ:  {self.__format__('.1e')}")
        return "\n".join(lines)
