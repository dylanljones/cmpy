# -*- coding: utf-8 -*-
"""
Created on 29 Mar 2019
author: Dylan

project: cmpy2
version: 1.0
"""
import numpy as np
from sciutils import Plot, eta, prange, Matrix
from cmpy.hubbard import HubbardModel, State, Siam

#  https://github.com/DerWeh/gftools/blob/master/gftools/__init__.py


def bethe_gf_omega(z, half_bandwidth=1):
    """Local Green's function of Bethe lattice for infinite Coordination number.
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
    TODO: source
    """
    z_rel = z / half_bandwidth
    return 2./half_bandwidth*z_rel*(1 - np.sqrt(1 - 1/(z_rel*z_rel)))


def bath_greens_function_inv(gf_loc, sigma=0):
    return 1/gf_loc + sigma


def bath_greens_function(gf_loc, sigma=0):
    return 1/bath_greens_function_inv(gf_loc, sigma)


def dmft_step(omega, sigma):
    gf_loc = bethe_gf_omega(omega - sigma)
    gf_bath_inv = bath_greens_function_inv(gf_loc, sigma)

    gf_imp = 10*np.random.rand(1)[0] - 5

    sigma_imp = gf_bath_inv - 1 / gf_imp
    return sigma_imp


def cost(gf_bath_inv, gf_0_inv, weights=1):
    arg = np.abs(gf_bath_inv - gf_0_inv) ** 2
    return np.sum(weights * arg) / arg.shape[0]


class NelderMeadSimplex:
    """ Nelder–Mead Simplex Solver

    See Also
    ========
    J. C. Lagarias, J. A. Reeds, M. H. Wright, P. E. Wright, Convergence Properties
    of the Nelder–Mead Simplex Method in Low Dimensions, SIAM J. Optim. 9 (1) (1998) 112–147.

    """

    def __init__(self, gf_bath_inv, omegas, rho=1, chi=2, gamma=0.5, sigma=0.5):
        self.gf_bath_inv = gf_bath_inv
        self.omegas = omegas
        self.rho = rho      # Reflection
        self.chi = chi      # Expansion
        self.gamma = gamma  # Contraction
        self.sigma = sigma  # Shrinkage

        self.result = None

    def f(self, x):
        siam = Siam(eps=x[0], v=x[1])
        gf_0_inv = self.omegas + siam.hybridization(self.omegas)
        return cost(self.gf_bath_inv, gf_0_inv)

    def iterstep(self, x):
        n_points = len(x)
        values = np.zeros(n_points)
        for i in range(n_points):
            values[i] = self.f(x[i])

        idx = np.argsort(values)
        x = x[idx]
        centroid = np.mean(x[:-1], axis=0)

        # Calculate reflection point
        x_r = (1+self.rho) * centroid - self.rho * x[-1]
        f_r = self.f(x_r)
        # Accept reflection point if x_0 <= x_r < x_{n+1}
        if values[0] <= f_r < values[-1]:
            self.result = x_r
            return None

        if f_r < values[0]:
            # Expand
            x_e = (1 + self.rho * self.chi) * centroid - self.rho * self.chi * x[-1]
            f_e = self.f(x_e)
            self.result = x_e if f_e < f_r else x_r
            return None

        elif values[-2] <= f_r < values[-1]:
            # Contract outside
            x_c = (1 + self.rho * self.gamma) * centroid - self.rho * self.gamma * x[-1]
            f_c = self.f(x_c)
            if f_c <= f_r:
                self.result = x_c
                return None

        elif f_r > values[-1]:
            # Contract inside
            x_c = (1 - self.gamma) * centroid + self.gamma * x[-1]
            f_c = self.f(x_c)
            if f_c < values[-1]:
                self.result = x_c
                return None

        # Perform shrink step
        x[1:] = x[0] + self.sigma * (x[1:] - x[0])
        return x


def main():
    n = 1000
    omega = np.linspace(-10, 10, n) + eta
    sigma = 0
    mu = 0
    n_b = 1
    n_p = 10000
    params = np.zeros((2, n_b))

    for i in range(100):
        gf_loc = bethe_gf_omega(omega + mu - sigma)
        gf_bath_inv = bath_greens_function_inv(gf_loc, sigma)

        nm = NelderMeadSimplex(gf_bath_inv, omega)

        x = params + np.random.uniform(-0.1, 0.1, size=(n_p, 2, n_b))
        while x is not None:
            x = nm.iterstep(x)

        params = nm.result
        eps, v = params

        siam = Siam(eps=eps, v=v, mu=mu)
        gf_0_inv = omega + siam.mu + siam.hybridization(omega)

        ham = siam.hamiltonian()
        gf = np.sum(ham.gf(omega), axis=1)
        gf_inv = 1/gf
        sigma_new = gf_0_inv - gf_inv
        delta = np.sum(np.abs(sigma - sigma_new)) / n
        print(delta)
        if delta < 0.01:
            break
        sigma = sigma_new

    gf_loc = bethe_gf_omega(omega + mu - sigma)
    plot = Plot()
    plot.plot(omega.real, gf_loc.imag)
    plot.show()


if __name__ == "__main__":
    main()
