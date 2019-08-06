# -*- coding: utf-8 -*-
"""
Created on 29 Mar 2019
author: Dylan

project: cmpy2
version: 1.0
"""
import numpy as np
from sciutils import Plot, eta
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



def bath_greens_function(gf_loc, sigma):
    gf_inv = 1/gf_loc + sigma
    return 1/gf_inv


def dmft():
    model = Siam(eps_d=0., u=10, eps=0., v=1., mu=0)

    n = 100
    omega = eta
    sigma = np.zeros(n)

    weiss = 1
    weiss_inv = 1/weiss
    gf_loc = bethe_gf_omega(omega)

    for i in range(100):
        sigma = weiss_inv - 1/gf_loc
        gf_loc = bethe_gf_omega(omega - sigma)
        weiss_inv = 1/gf_loc + sigma


def hf_qmc(gf_bath, t=0.05, nt=100):
    beta = 1/t
    delta_tau = beta/nt





def main(sigma_0=1, nt=100):
    sigma = sigma_0
    omega = eta
    gf_loc = bethe_gf_omega(omega - sigma)
    gf_bath = bath_greens_function(gf_loc, sigma)
    hf_qmc(gf_bath)
    print(gf_loc, gf_bath)










if __name__ == "__main__":
    main()
