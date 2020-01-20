# -*- coding: utf-8 -*-
"""
Created on 11 Nov 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
import scipy.linalg as la

# Pauli matrices
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])


def vlinspace(start, stop, n=1000):
    """ Vector linspace

    Parameters
    ----------
    start: array_like
        d-dimensional start-point
    stop: array_like
        d-dimensional stop-point
    n: int, optional
        number of points, default=1000

    Returns
    -------
    vectors: np.ndarray
    """
    axes = [np.linspace(start[i], stop[i], n) for i in range(len(start))]
    return np.asarray(axes).T


def chain(items, cycle=False):
    """ Create chain between items

    Parameters
    ----------
    items: array_like
        items to join to chain
    cycle: bool, optional
        cycle to the start of the chain if True, default: False

    Returns
    -------
    chain: list
        chain of items

    Example
    -------
    >>> print(chain(["x", "y", "z"]))
    [['x', 'y'], ['y', 'z']]

    >>> print(chain(["x", "y", "z"], True))
    [['x', 'y'], ['y', 'z'], ['z', 'x']]
    """
    result = list()
    for i in range(len(items)-1):
        result.append([items[i], items[i+1]])
    if cycle:
        result.append([items[-1], items[0]])
    return result


def normalize(array):
    """ Normalizes a given array

    Parameters
    ----------
    array: array_like
        Un-normalized array

    Returns
    -------
    arr_normalized: np.ndarray
    """
    return np.asarray(array) / np.linalg.norm(array, ord=1)


def update_mean(mean, num, x):
    """ Calculate average iteratively

    Parameters
    ----------
    mean: float
        old mean-value
    num: int
        number of iteration
    x: float
        new value

    Returns
    -------
    mena: float
    """
    return mean + (x - mean) / num


def distance(r1, r2):
    """ calculate distance bewteen two points
    Parameters
    ----------
    r1: array_like
        first point
    r2: array_like
        second point

    Returns
    -------
    distance: float
    """
    diff = np.abs(np.asarray(r1) - np.asarray(r2))
    return np.sqrt(np.dot(diff, diff))


def check_hermitian(a, tol=1e-8):
    r""" Checks if the given matrix is hermitian

    .. math::
        A = A^\dagger \Longleftrightarrow A - A^\dagger < tol

    Parameters
    ----------
    a: (N, N) ndarray
        Matrix to check for hermiticity.
    tol: float
        Tolerance for check.

    Returns
    -------
    is_hermitian: bool
    """
    return np.all(np.abs(a-a.conj().T) <= tol)


def decompose(a, sym_tol=1e-8):
    r""" Decomposes the matrix into it's eigenvalues and eigenvectors.

    If the given matrix is hermitian, the inverse of the right eigenvectors
    is the adjoint :math'U^\dagger':

    .. math::
        A = V X V^\dagger, \quad V = diag(λ(A))

    Otherwise the inverse is calculated:

    .. math::
        A = V X V^{-1}, \quad V = diag(λ(A))

    Parameters
    ----------
    a: (N, N) np.ndarray
        matrix to decompose
    sym_tol: float
        Tolerance for checking if matrix is hermitian

    Returns
    -------
    eigvecs_inv: (N, N) complex ndarray
        The inverse or adjunct of the right eigenvector
    eigvals: (N) float ndarray
        The eigenvalues of the matrix
    eigvecs: (N, N) complex ndarray
        The right eigenvectors of the matrix
    """
    eigvals, eigvecs = np.linalg.eig(a)
    if check_hermitian(a, sym_tol):
        eigvecs_inv = eigvecs.conj().T
    else:
        eigvecs_inv = la.inv(eigvecs)
    return eigvecs_inv, eigvals, eigvecs


def reconstruct(v_inv, x, v):
    r"""Reconstruct a  matrix from it's eigendecomposition.

    .. math::
        A = V X V^{-1}, \quad V = diag(λ(A))

    v_inv: (N, N) complex ndarray
        The inverse or adjunct of the right eigenvector
    x: (N) float ndarray
        The eigenvalues of the matrix
    v: (N, N) complex ndarray
        The right eigenvectors of the matrix

    Returns
    -------
    a: (N, N) ndarray:
        Reconstructed matrix
    """
    return np.round(np.matmul(v * x, v_inv), decimals=10)


def partition(energies, beta):
    r""" Calculates the canonical partition function

    .. math::
        Z = \sum_i e^{-\beta E_i}

    Parameters
    ----------
    energies: (N) ndarray
        Energie values of the microstates :math'i'.
    beta: float
        Thermodynamic beta: :math'\beta=1/k_B T' (inverse temperature).

    Returns
    -------
    z: float
        Canonical partition function.
    """
    return np.exp(-beta*energies).sum()


def fermi_dist(energy, beta=None, mu=0.):
    r""" Calculates the fermi-distributions for fermions.

    ..math ::
        f(\epsilon) = \frac{1}{e^{\beta (\epsilon - \mu)} + 1}

    Parameters
    ----------
    energy: ndarray or float
        The energy value :math'\epsilon'.
    beta: float, optional
        Thermodynamic beta: :math'\beta=1/k_B T' (inverse temperature).
        The default is 'None', corresponding to .math:'T=0'
    mu: float, optional
        Chemical potential :math'\mu=0'. At :math'T=0' this is the Fermi-energy :math'\epsilon_F'.
        The default is :math'\mu=0'.

    Returns
    -------
    fermi: (N) ndarray
        Fermi distribuiton.
    """
    if beta is None:
        f = np.zeros_like(energy)
        f[energy <= mu] = 1.
    else:
        exponent = np.asarray((energy - mu) * beta).clip(-1000, 1000)
        f = 1. / (np.exp(exponent) + 1)
    return f


def diagonalize(operator):
    """diagonalizes single site Spin Hamiltonian"""
    eig_values, eig_vecs = la.eigh(operator)
    eig_values -= np.amin(eig_values)
    return eig_values, eig_vecs


def expectation(eigvals, eigstates, operator, beta):
    ew = np.exp(-beta * eigvals)
    aux = np.einsum('i,ji,ji', ew, eigstates, operator.dot(eigstates))
    return aux / ew.sum()


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
