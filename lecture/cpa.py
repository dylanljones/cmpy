# coding: utf-8
#
# This code is part of cmpy.
# 
# Copyright (c) 2021, Dylan Jones

import numpy as np
from numpy.lib import scimath
from scipy import linalg as la
import matplotlib.pyplot as plt
from cmpy.collection import get_bins
from abc import abstractmethod, ABC


def gf0_z_onedim(z, half_bandwidth):
    z_rel_inv = half_bandwidth / z
    return 1. / half_bandwidth * z_rel_inv / scimath.sqrt(1 - z_rel_inv ** 2)


def hilbert(xi, half_bandwidth):
    return gf0_z_onedim(xi, half_bandwidth)


def sample_eigvals(num_sites, con, eps, t=1.0, samples=10, bin_size=20):
    num_sites = int(num_sites)
    vals = np.array(eps)
    hopp = np.full(num_sites - 1, fill_value=t)
    eigvals_hist = list()
    edges = None
    print("Sampling....", end="", flush=True)
    for i in range(samples):
        print(f"\rSampling {i+1}/{samples}", end="", flush=True)
        eps_rand = np.random.choice(vals, size=num_sites, p=con)
        eigvals = la.eigvalsh_tridiagonal(eps_rand, hopp)
        hist, edges = np.histogram(eigvals, bins=num_sites//bin_size, density=True)
        eigvals_hist.append(hist)
    print()
    bins = get_bins(edges, 0.5)
    hist = np.array(eigvals_hist)
    return bins, hist


def percentile(hist, p=50., axis=0):
    return np.percentile(hist, p, axis=axis)


def histogram_median(hist, dp=31.7/2):
    median = percentile(hist, 50)
    hist_up = percentile(hist, 100 - dp)
    hist_dn = percentile(hist, dp)
    return median, np.abs(median - [hist_dn, hist_up])


class SingleSiteApproximation(ABC):

    def __init__(self, energies, concentrations, hop=1., label=""):
        assert len(energies) == len(concentrations)
        self.eps = np.array(energies)
        self.con = np.array(concentrations)
        self.half_bandwith = 2 * hop
        self.label = label or self.__class__.__name__

    @abstractmethod
    def greens(self, z) -> np.array:
        """Computes the Green's function"""
        pass

    def dos(self, z):
        return -self.greens(z).imag / np.pi


class VCA(SingleSiteApproximation):

    def __init__(self, energies, concentrations, hop=1.):
        super().__init__(energies, concentrations, hop)

    def greens(self, z) -> np.array:
        """Computes the Green's function in the virtual crystal approximation."""
        sigma = sum(self.eps * self.con)
        return gf0_z_onedim(z - sigma, self.half_bandwith)


class ATA(SingleSiteApproximation):

    def __init__(self, energies, concentrations, hop=1.):
        super().__init__(energies, concentrations, hop)

    def greens(self, z) -> np.array:
        """Computes the Green's function in the average T-matrix approximation."""

        # Unperturbated Green's function (uses VCA)
        sigma_vca = sum(self.eps * self.con)
        gf0 = gf0_z_onedim(z - sigma_vca, self.half_bandwith)

        # Average T-matrix
        tavrg_a = self.con[0] * (self.eps[0] - sigma_vca) / (1 - (self.eps[0] - sigma_vca) * gf0)
        tavrg_b = self.con[1] * (self.eps[1] - sigma_vca) / (1 - (self.eps[1] - sigma_vca) * gf0)
        tavrg = tavrg_a + tavrg_b

        # Self energy
        sigma = tavrg / (1 + gf0 * tavrg)

        return gf0_z_onedim(z - sigma_vca - sigma, self.half_bandwith)


class CPA(SingleSiteApproximation):

    def __init__(self, energies, concentrations, hop=1., thresh=1e-3, maxiter=1000):
        super().__init__(energies, concentrations, hop)
        self.maxiter = maxiter
        self.thresh = thresh

    def greens(self, z) -> np.array:
        """Computes the Green's function in the coherent potential approximation."""
        sigma = np.zeros(len(z))
        sigma_old = np.inf
        gf_avrg = np.zeros_like(z)
        for i in range(self.maxiter):
            # Compute average GF via the self-energy
            gf_avrg = gf0_z_onedim(z - sigma, self.half_bandwith)
            gf0_inv = sigma + 1 / gf_avrg

            # Dyson equation
            gf_a = 1 / (sigma - self.eps[0] + 1 / gf_avrg)
            gf_b = 1 / (sigma - self.eps[1] + 1 / gf_avrg)

            gf_avrg = self.con[0] * gf_a + self.con[1] * gf_b
            sigma = gf0_inv - 1 / gf_avrg

            diff = np.trapz(abs(sigma - sigma_old), z.real)
            if diff < self.thresh:
                print(f"Converged in {i} iterations")
                break
            sigma_old = sigma
        else:
            print("CPA self consistency not reached")

        return gf_avrg


def main():
    num_sites = 1e4
    t = 1.
    c = 0.1

    con = np.array([c, 1 - c])
    eps = np.array([-1, +1])
    omega = np.linspace(-4, +4, 1000)
    z = omega + 1e-4j

    bins, hist = sample_eigvals(num_sites, con, eps, t, samples=20)
    hist_median, hist_errs = histogram_median(hist)

    vca = VCA(eps, con, t)
    ata = ATA(eps, con, t)
    cpa = CPA(eps, con, t)

    fig, ax = plt.subplots(figsize=(5 * 16 / 9, 5))
    ax.errorbar(bins, hist_median, yerr=hist_errs, ecolor='red', label="EIG")
    ax.plot(z.real, vca.dos(z), label=vca.label, lw=1.5)
    ax.plot(z.real, ata.dos(z), label=ata.label, lw=1.5)
    ax.plot(z.real, cpa.dos(z), label=cpa.label, lw=1.5)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
