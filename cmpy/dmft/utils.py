# coding: utf-8
#
# This code is part of cmpy.
# 
# Copyright (c) 2021, Dylan Jones

import numpy as np
import matplotlib.pyplot as plt
from collections import Sequence
from typing import List
import colorcet as cm
from colorama import Fore


def plot_dos_contour(z, u, dos, title="", fig=None, ax=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots()
    xx, yy = np.meshgrid(z.real, u)
    surf = ax.contourf(xx, yy, dos, cmap=cm.m_blues, vmin=0)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$U$")
    ax.set_ylim(0, np.max(yy))
    ax.grid()
    if title:
        ax.set_title(title)
    fig.colorbar(surf, label=r"$-iG(\omega)$")
    fig.tight_layout()
    return fig, ax


def plot_dos(z, dos, u, title="", fig=None, ax=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots()
    x = z.real
    ax.set_title(f"U={u:.2f}")
    ax.plot(x, dos)
    ax.fill_between(x, 0, dos, color="C0", alpha=0.5)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho(\omega)$")
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(0, ax.get_ylim()[1])
    if title:
        ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid()
    fig.tight_layout()
    return fig, ax


def plot_quasiparticle_weight(u, qp_weight, title="", fig=None, ax=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots()
    ax.plot(u, qp_weight)
    ax.set_xlabel(r"$U$")
    ax.set_ylabel(r"$Z$")
    ax.set_xlim(np.min(u), np.max(u))
    ax.set_ylim(0, ax.get_ylim()[1])
    if title:
        ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid()
    fig.tight_layout()
    return fig, ax


class IterationStats(Sequence):

    def __init__(self, *labels):
        super().__init__()
        self._errors = list()
        self.labels = labels
        self.success = False
        self.status = ""

    @property
    def iteration(self):
        return len(self._errors)

    @property
    def errors(self):
        return self._errors[-1]

    @property
    def history(self):
        return np.asarray(self._errors).T

    @property
    def message(self):
        if self.success:
            msg = "Self-consistency reached"
        else:
            msg = "Self-consistency not reached"
        return msg + ": " + self.status

    def append(self, *items):
        self._errors.append(list(items))

    def set_status(self, success, status=""):
        self.success = success
        self.status = status

    def set_maxiter_status(self, maxiter):
        self.success = False
        self.status = f"Maximum iteration of {maxiter} reached without converging!"

    def set_parameter_converged(self, name, value, dec=4):
        self.success = True
        self.status = f"{name} converged to {value:.{dec}f}"

    def plot(self, show=True, grid="both", xlabel="Iteration", ylabel="Error",
             fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        arrays = self.history
        labels = self.labels or [None for _ in range(arrays.shape[1])]
        for arr, label in zip(arrays, labels):
            ax.plot(arr, label=label)
        if any(labels):
            ax.legend()
        ax.set_xlim(0, self.iteration)
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(which=grid)
        if show:
            plt.show()
        return fig, ax

    def __bool__(self):
        return self.success

    def __len__(self) -> int:
        return len(self._errors)

    def __iter__(self):
        yield from np.atleast_1d(self.errors)

    def __getitem__(self, i: int) -> List[float]:
        return self._errors[i]

    def errorstr(self, frmt=".2e"):
        return ", ".join([f"{label}={err:{frmt}}" for label, err in zip(self.labels, self.errors)])

    def __str__(self):
        reset = Fore.RESET
        col = "" if self.success else Fore.RED
        lines = list()
        lines.append(f"Success:    {self.success}")
        lines.append(f"Iterations: {self.iteration}")
        lines.append(f"Errors:     " + self.errorstr(".1e"))
        lines.append(f"Status:     {col + self.status + reset}")
        return "\n".join(lines)


def mix_values(old, new, mixing=1.):
    """Update a parameter using mixing.

    Parameters
    ----------
    old : array_like
        The old value of the parameter.
    new : array_like
        The new value of the parameter.
    mixing : float, optional
        The amount of mixing to use. If a `1` is passed no mixing is applied
        and the new value is returned unmodified.

    Returns
    -------
    new_mixed: array_like
        The mixed new value.
    """
    if mixing == 1:
        return new
    assert 0 < mixing < 1

    return new * mixing + old * (1.0 - mixing)


def self_energy(gf_imp0: (float, np.ndarray),
                gf_imp: (float, np.ndarray)) -> (float, np.ndarray):
    r"""Computes the self energy .math:`\Sigma(z)`.

    The self energy is defined as:
    ..math::
        \Sigma(z) = G_0(z)^{-1} - G(z)^{-1}
    where .math:`G_0(z)` and .math:`G(z)` are the non-interacting and interacting
    Green's function.

    Parameters
    ----------
    gf_imp0: float or (N) np.ndarray
        The non-interacting Green's function .math:`G_0`.
    gf_imp: float or (N) np.ndarray
        The interacting Green's function .math:`G`.

    Returns
    -------
    sigma: float or (N) np.ndarray
        The array containing the self energy.
    """
    return 1/gf_imp0 - 1/gf_imp


def bethe_gf_omega(z: (complex, np.ndarray), t: float) -> (complex, np.ndarray):
    """Local Green's function of the Bethe lattice for infinite Coordination number.

    Parameters
    ----------
    z: complex or (N) complex ndarray
        Complex frequency `z`
    t: float
        The hopping parameter of the lattice-model.

    Returns
    -------
    gf_omega : complex or (N) complex ndarray
        Value of the Green's function
    """
    z_rel = z / (2 * t)
    return z_rel * (1 - np.sqrt(1 - 1 / (z_rel * z_rel))) / t


def quasiparticle_weight(omegas: np.ndarray, sigma: np.ndarray,
                         thresh: float = 1e-5) -> np.ndarray:
    r"""Computes the quasiparticle .math:`z_{qp}` weight.

    The quasiparticle weight is defined via the derivative of the self energy
    at .math:`\omega = 0`:
    ..math::
        z_{qp} = \left[ 1 - \frac{d\Sigma(0)}{\omega} \right]^{-1}

    References
    ----------
    'Two-site dynamical mean-field theory' by M. Potthof:
    https://arxiv.org/abs/cond-mat/0107502

    Parameters
    ----------
    omegas: (N) np.ndarray
        The real frequency .math:`\omega`.
    sigma: (N) np.ndarray
        The self energiy .math:`\Sigma(\omega)`.
    thresh: float, optional
        The quasiparticle weight is set to zero if the value falls
        below the threshold.

    Returns
    -------
    z_qp: (N) np.ndarray
        Array of the quasiparticle weights.
    """
    dw = omegas[1] - omegas[0]
    win = (-dw <= omegas) * (omegas <= +dw)
    try:
        dsigma = np.polyfit(omegas[win], sigma.real[win], 1)[0]
        z_qp = 1 / (1 - dsigma)
        if z_qp < thresh:
            z_qp = 0
    except np.linalg.LinAlgError:
        # weird linalg/OPENBLAS error: https://github.com/numpy/numpy/issues/16744
        # only occurs for low temps / high betas
        z_qp = 0
    return z_qp
