# -*- coding: utf-8 -*-
"""
Created on 19 Sep 2019
author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
import scipy.linalg as la
from scipy import integrate
from scipy import optimize
from scipy.sparse import csr_matrix
from itertools import product
import matplotlib.pyplot as plt


def bethe_dos(z, t):
    """Density of states of the Bethe lattice"""
    energy = np.asarray(z).clip(-2 * t, 2 * t)
    return np.sqrt(4 * t**2 - energy**2) / (2 * np.pi * t**2)


def bethe_gf_omega(z, t=1.0):
    """Local Green's function of Bethe lattice for infinite Coordination number.

    Taken from gf_tools by Weh Andreas
    https://github.com/DerWeh/gftools/blob/master/gftools/__init__.py

    Parameters
    ----------
    z : complex ndarray or complex
        Green's function is evaluated at complex frequency `z`
    t : float
        Hopping parameter of the bethe lattice. This defines the bandwidth 'D=4t'
    Returns
    -------
    bethe_gf_omega : complex ndarray or complex
        Value of the Green's function
    """
    half_bandwidth = 2 * t
    z_rel = z / half_bandwidth
    return 2. / half_bandwidth * z_rel * (1 - np.sqrt(1 - 1 / (z_rel * z_rel)))


# =========================================================================
# Basis states and operators
# =========================================================================


def basis_states(sites):
    """ Creates basis states for a many-body system in binary representation

    The states are initialized as integer. The binary represents the occupation of the lattice.

    idx     ↓3 ↑3 ↓2 ↑2 ↓1 ↑1
    binary  0  1  0  1  1  0

    Parameters
    ----------
    sites: int
        Number of sites in the system
    Returns
    -------
    states: list of int
    """
    n = 2 ** (2 * sites)
    return list(range(n))


def annihilate(state, i):
    """ Act annihilation operator on state

    Parameters
    ----------
    state: int
        Many-body state in binary representation
    i: int
        Index of annihilation operator

    Returns
    -------
    state: int
        Annihilated state
    """
    if not int(state >> i) & 1:
        return None
    return state ^ (1 << i)


def phase(state, i):
    """Phase for fermionic operators"""
    particles = bin(state >> i + 1).count("1")
    return 1 if particles % 2 == 0 else -1


def annihilation_operator(states, idx):
    """ Create annihilation operator in matrix representation for a given set of basis states

    Parameters
    ----------
    states: list_like of int
        Basis states of the system
    idx: int
        Index of annihilation operator

    Returns
    -------

    """
    n = len(states)
    row, col, data = list(), list(), list()
    for j in range(n):
        state = states[j]
        other = annihilate(state, idx)
        if other is not None:
            i = states.index(other)
            val = phase(state, idx)
            row.append(i)
            col.append(j)
            data.append(val)
    return csr_matrix((data, (row, col)), shape=(n, n), dtype="int")


# =========================================================================
# Green's function and impurity model
# =========================================================================

def diagonalize(operator):
    """ Diagonalizes the given operator"""
    eig_values, eig_vecs = la.eigh(operator)
    # eig_values -= np.amin(eig_values)
    return eig_values, eig_vecs


def greens_function(eigvals, eigstates, operator, z, beta=1.):
    """ Calculate the interacting Green's function in Lehmann representation

    Parameters
    ----------
    eigvals: array_like
        Eigenvalues of the many-body system
    eigstates: array_like
        Eigenvectors of the many-body system
    operator: array_like
        Annihilation operator in matrix representation of a given site and spin
    z: array_like
        Energy values to evaluate the Green's function
    beta: float
        Inverse of the temperature

    Returns
    -------
    gf: np.ndarray
    """

    # Create basis and braket matrix <n|c|m> of the given operator
    basis = np.dot(eigstates.T, operator.dot(eigstates))
    qmat = np.square(basis)

    # Calculate the energy gap matrix
    gap = np.add.outer(-eigvals, eigvals)

    # Calculate weights and partition function
    ew = np.exp(-beta*eigvals)
    weights = np.add.outer(ew, ew)
    partition = ew.sum()

    # Construct Green's function
    n = eigvals.size
    gf = np.zeros_like(z)
    for i, j in product(range(n), range(n)):
        gf += qmat[i, j] / (z - gap[i, j]) * weights[i, j]
    return gf / partition


def self_energy(gf_imp0, gf_imp):
    """ Calculate the self energy from the non-interacting and interacting Green's function"""
    return 1/gf_imp0 - 1/gf_imp


def m2_weight(t):
    """ Calculates the second moment weight"""
    return integrate.quad(lambda x: x*x * bethe_dos(x, t), -2*t, 2*t)[0]


def quasiparticle_weight(omegas, sigma):
    """ Calculates the quasiparticle weight"""
    dw = omegas[1] - omegas[0]
    win = (-dw <= omegas) * (omegas <= dw)
    dsigma = np.polyfit(omegas[win], sigma.real[win], 1)[0]
    z = 1/(1 - dsigma)
    if z < 0.01:
        z = 0
    return z


def filling(omegas, gf):
    """ Calculate the filling using the Green's function of the corresponding model"""
    idx = np.argmin(np.abs(omegas)) + 1
    x = omegas[:idx]
    y = -gf[:idx].imag
    x[-1] = 0
    y[-1] = (y[-1] + y[-2]) / 2
    return integrate.simps(y, x)


class HamiltonOperator:

    def __init__(self, operators):
        c0u, c0d, c1u, c1d = operators
        self.u_op = c0u.T * c0u * c0d.T * c0d
        self.eps_imp_op = c0u.T * c0u + c0d.T * c0d
        self.eps_bath_op = c1u.T * c1u + c1d.T * c1d
        self.v_op = (c0u.T * c1u + c1u.T * c0u) + (c0d.T * c1d + c1d.T * c0d)

    def build(self, u=5., eps_imp=0., eps_bath=0., v=1.):
        return u * self.u_op + eps_imp * self.eps_imp_op + eps_bath * self.eps_bath_op + v * np.abs(self.v_op)


class TwoSiteSiam:

    def __init__(self, u, eps_imp, eps_bath, v, mu, beta=0.01):
        self.mu = mu
        self.beta = beta
        self.u = float(u)
        self.eps_imp = float(eps_imp)
        self.eps_bath = float(eps_bath)
        self.v = float(v)

        self.states = basis_states(2)
        self.ops = [annihilation_operator(self.states, i) for i in range(4)]
        self.ham_op = HamiltonOperator(self.ops)
        self.eig = None

    def update_bath_energy(self, eps_bath):
        self.eps_bath = float(eps_bath)

    def update_hybridization(self, v):
        self.v = float(v)

    def param_str(self, dec=2):
        u = f"u={self.u:.{dec}}"
        eps_imp = f"eps_imp={self.eps_imp:.{dec}}"
        eps_bath = f"eps_bath={self.eps_bath:.{dec}}"
        v = f"v={self.v:.{dec}}"
        return ", ".join([u, eps_imp, eps_bath, v])

    def bathparam_str(self, dec=2):
        eps_bath = f"eps_bath={self.eps_bath:.{dec}}"
        v = f"v={self.v:.{dec}}"
        return ", ".join([eps_bath, v])

    def hybridization(self, z):
        delta = self.v**2 / (z + self.mu - self.eps_bath)
        return delta

    def hamiltonian(self):
        return self.ham_op.build(self.u, self.eps_imp, self.eps_bath, self.v)

    def diagonalize(self):
        ham_sparse = self.hamiltonian()
        self.eig = diagonalize(ham_sparse.todense())

    def impurity_gf(self, z, spin=0):
        self.diagonalize()
        eigvals, eigstates = self.eig
        return greens_function(eigvals, eigstates, self.ops[spin].todense(), z + self.mu, self.beta)

    def impurity_gf_free(self, z):
        return 1/(z + self.mu - self.eps_imp - self.hybridization(z))

    def __str__(self):
        return f"Siam({self.param_str()})"


class TwoSiteDmft:

    def __init__(self, z, u=5, eps=0, t=1, mu=None, eps_bath=None, beta=10.):
        self.mu = u / 2 if mu is None else mu
        self.z = z
        self.t = t
        eps_bath = mu if eps_bath is None else eps_bath
        self.siam = TwoSiteSiam(u, eps, eps_bath, t, mu, beta)
        self.m2 = t  # m2_weight(t)

        self.gf_imp0 = None
        self.gf_imp = None
        self.sigma = None
        self.gf_latt = None
        self.quasiparticle_weight = None

    def solve(self, spin=0):
        self.gf_imp0 = self.siam.impurity_gf_free(self.z)
        self.gf_imp = self.siam.impurity_gf(self.z, spin=spin)
        self.sigma = self_energy(self.gf_imp0, self.gf_imp)
        self.gf_latt = bethe_gf_omega(self.z + self.mu - self.sigma, self.t)
        self.quasiparticle_weight = quasiparticle_weight(self.z.real, self.sigma)

    def impurity_filling(self):
        return filling(self.z.real, self.gf_imp) / np.pi

    def lattice_filling(self):
        return filling(self.z.real, self.gf_latt) / np.pi

    def filling_condition(self, eps_bath):
        self.siam.update_bath_energy(eps_bath)
        self.solve()
        return self.impurity_filling() - self.lattice_filling()

    def optimize_filling(self, tol=1e-2):
        x = np.asarray([self.siam.eps_bath])
        sol = optimize.root(self.filling_condition, x0=x, tol=tol)
        if not sol.success:
            raise ValueError(f"Failed to optimize filling! ({self.siam.param_str()})")
        else:
            self.siam.update_bath_energy(sol.x)
            self.solve()
            return self.impurity_filling() - self.lattice_filling()

    def new_hybridization(self, mixing=1.0):
        z = self.quasiparticle_weight
        v_new = np.sqrt(z * self.m2)
        new, old = mixing, 1.0 - mixing
        return (v_new * new) + (self.siam.v * old)

    def solve_self_consistent(self, spin=0, mixing=1.0, vtol=1e-4, ntol=1e-2, nmax=1000):
        v = self.siam.v + 1e-10
        it, delta_v, delta_n = 0, 0, 0
        for it in range(nmax):
            # self.optimize_filling(ntol)
            self.siam.update_hybridization(v)
            self.solve(spin)
            if self.quasiparticle_weight == 0:
                break
            v_new = self.new_hybridization(mixing)
            delta_v = abs(v - v_new)
            v = v_new
            if delta_v <= vtol:
                break

        self.siam.update_hybridization(v)
        self.solve()
        return it, delta_v, delta_n


def plot_lattice_dos(z, u, eps, t, mu, eps_bath=0):
    dmft = TwoSiteDmft(z, u, eps, t, mu, eps_bath)
    it, delta_v, delta_n = dmft.solve_self_consistent()
    print(f"Self-consistency reached: {dmft.siam.bathparam_str()}")
    print(f"It: {it}, Delta_v: {delta_v:.1e}, Delta_n: {delta_n:.1e}")

    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$A_{latt}$")
    ax.plot(dmft.z.real, -dmft.gf_latt.imag)
    # ax.plot(dmft.z.real, -dmft.sigma.imag, color="k", ls="--")
    plt.show()


def quasiparticle_line(z, eps, t, umax=8, n=20, beta=0.01):
    u_values = np.linspace(0, umax, n)
    qp_weights = np.zeros_like(u_values)
    for i, u in enumerate(u_values):
        # print(f"U={u}")
        mu = u/2
        dmft = TwoSiteDmft(z, u, eps, t, mu, eps_bath=mu, beta=beta)
        it, delta_v, _ = dmft.solve_self_consistent()
        qp_weights[i] = dmft.quasiparticle_weight
        print(f"U={u:.1f}  {it} Delta: {delta_v:.2e}")
    return u_values, qp_weights


def plot_quasiparticle_weight(z, eps, t, umax=8, n=20, betas=(0.01, )):
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$U$")
    ax.set_ylabel(r"$z$")
    ax.set_xlim(0, umax)
    ax.set_ylim(0, 1)
    for beta in betas:
        u_values, qp_weights = quasiparticle_line(z, eps, t, umax, n, beta=beta)
        ax.plot(u_values, qp_weights, label=f"{beta}")
        print()
    ax.legend()
    plt.show()


def main():
    u = 0.1
    eps, t = 0, 1
    mu = u/2
    omegas = np.linspace(-10, 10, 10000)
    z = omegas + 0.01j

    # plot_lattice_dos(z, u, eps, t, mu, eps_bath=mu)
    plot_quasiparticle_weight(z, eps, t, n=40, umax=8, betas=(0.001, 0.1, 10))


if __name__ == "__main__":
    main()
