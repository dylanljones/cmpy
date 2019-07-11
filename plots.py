# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2018
@author: Dylan Jones

project: plots$
version: 1.0
"""
import os
import time
import numpy as np
from scipy import linalg as la
from cmpy2 import prange, Progress, greens, DATA_DIR, Folder
from cmpy2.core.plotting import Plot, plot_banddos, build_bands
from cmpy2.tightbinding import TbDevice, p3_basis, s_basis
from cmpy2.tightbinding import LT_Data, loc_length, loc_length_fit
from cmpy2 import eta

DPI = 600
POINTS = [0, 0], [np.pi, 0], [np.pi, np.pi]
NAMES = r"$\Gamma$", r"$X$", r"$M$"

SCRIPT = r"D:\Dropbox\Work\Latex_Doc\img"

# =========================================================================
# Tight-binding theory
# =========================================================================


def plot_bands():
    k = np.linspace(-np.pi, np.pi, 1000)
    band = 2 * np.cos(k)

    x = band/2
    klim = x[0], k[-1]
    plot = Plot()
    plot.set_labels(r"$k$", r"$\epsilon[t]$")
    plot.set_limits(klim, [-2.1, 2.1])
    plot.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], np.arange(-2, 3))
    plot.set_ticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

    plot.plot(k, band)
    plot.format_latex(textwidth=0.6)
    plot.save(SCRIPT, "tb_1d_chain.eps")
    # plot.show()


def plot_dimer_dos():
    model = TbDevice.square((2, 1))
    ham = model.hamiltonian()

    n = 1000
    omegas = np.linspace(-2, 2, n)
    dos = np.zeros(n)
    for i in range(n):
        dos[i] = ham.dos(omegas[i] + 0.01j)

    plot = Plot()
    plot.set_labels(r"$(\omega-\epsilon)/ t$", r"$n(E)$")
    plot.set_limits((-2, 2))
    plot.set_ticks((-1, 0, 1))
    x = -1
    plot.ax.annotate("$E_0$", xy=(x - 0.1, 15), xytext=(x - 0.9, 20), arrowprops=dict(arrowstyle="->"))
    x = 1
    plot.ax.annotate("$E_1$", xy=(x - 0.1, 15), xytext=(x - 0.9, 20), arrowprops=dict(arrowstyle="->"))
    plot.plot(omegas, dos)

    plot.format_latex(textwidth=0.6)
    # plot.show()
    plot.save(SCRIPT, "dos_tbdimer.eps")


def fdos(e, eps=0, t=1):
    if abs(e) < 2*t:
        return 1 / (2 * np.pi * np.sqrt(1-((e-eps)/(2*t))**2))
    else:
        return 0


def plot_chain_dos():
    xmax = 3
    xlim = -xmax, xmax
    n = 500
    omegas = np.linspace(*xlim, n)
    plot = Plot(xlim=xlim, xlabel=r"$(\omega-\epsilon)/ t$", ylabel=r"$n(E)$")

    dos = np.zeros(n)
    for i in range(n):
        dos[i] = fdos(omegas[i])
    plot.plot(omegas, dos, label=r"$n_{ana}(E)$")

    size = 5
    center = 2
    model = TbDevice.square((size, 1))
    ham = model.hamiltonian()
    ham.set_hopping(0, 4, 1)

    for i in prange(n, header="Calculating GF"):
        gf = ham.greens(omegas[i] + 0.01j, only_diag=True)
        dos[i] = -1/np.pi * np.sum(gf[center].imag, axis=0)
    plot.plot(omegas, dos, color="k", lw=1, label=r"$n_{num}^{(" + str(size) + r")}(E)$")

    plot.legend()
    # plot.format_latex(textwidth=0.5, ratio=0.7)
    plot.show()
    # plot.save(SCRIPT, "dos_tb1d2.eps")


def plot_chain(textwidth=1., ratio=None):
    ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    tick_labels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
    emax = 3
    elim = -emax, emax
    ylim = -2.5, 2.5
    n = 1000
    size = 1000
    center = int(size / 2)

    k = np.linspace(-np.pi, np.pi, n)
    band = 2 * np.cos(k)

    x = band/2
    klim = x[0], k[-1]
    plot1 = Plot(xlim=klim, xlabel=r"$k$", ylim=ylim, ylabel=r"$\epsilon[t]$")
    plot1.set_ticks(ticks, np.arange(-2, 3))
    plot1.set_ticklabels(tick_labels)

    plot1.plot(k, band)

    # ========== DOS ==========

    omegas = np.linspace(*elim, n)
    plot2 = Plot(xlim=(-0.1, 1.5), ylim=ylim, ylabel=r"$(\omega-\epsilon)/ t$", xlabel=r"$n(E)$")
    plot1.lines(y=[-2, 2], color="0.5", lw=1)
    plot2.lines(y=[-2, 2], color="0.5", lw=1)

    dos = np.zeros(n)
    model = TbDevice.square((size, 1))
    ham = model.hamiltonian()
    for i in prange(n, header="Calculating GF"):
        gf = ham.greens(omegas[i] + 0.01j, only_diag=True)
        dos[i] = -1 / np.pi * np.sum(gf[center].imag, axis=0)
    plot2.plot(dos, omegas, color="k", lw=0.5, label=r"$n_{num}(E)$")
    for i in range(n):
        dos[i] = fdos(omegas[i])
    plot2.plot(dos, omegas, label=r"$n_{ana}(E)$")

    plot2.legend()
    plot1.format_latex(textwidth=textwidth, ratio=ratio)

    plot2.format_latex(textwidth=textwidth, ratio=ratio)

    # plot1.save(SCRIPT, "tbchain_a.eps")
    # plot2.save(SCRIPT, "tbchain_b.eps")

    plot2.show()


def plot_bandstruct():
    model = TbDevice.square()
    band_sections = model.bands(POINTS)

    point_names = list(NAMES)
    point_names.append(point_names[0])

    plot = Plot(xlabel=r"$k$", ylabel=r"$E(k)$")
    n_sections = len(band_sections)
    x0, ticks = 0, [0]
    color = "C0"
    for i in range(n_sections):
        section = band_sections[i]
        x = x0 + np.arange(section.shape[0])
        plot.plot(x, section, color=color)
        x0 = max(x) + 1
        ticks.append(x0)

    plot.set_limits(xlim=(0, ticks[-1]))
    plot.set_ticks(xticks=ticks)
    plot.set_ticklabels(xticks=point_names)
    plot.lines(y=0, lw=1, color="0.5")
    plot.lines(x=ticks[1:-1], lw=0.5, color="0.5")

    plot.format_latex(textwidth=0.7)
    # plot.save(SCRIPT, "tb2d_disp.eps")
    plot.show()


def plot_bandstruct_2d():
    ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    tick_labels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]

    n = 1000
    klim = -np.pi, np.pi
    x = np.linspace(*klim, n)
    y = np.linspace(*klim, n)
    xx, yy = np.meshgrid(x, y)
    zz = 2 * (np.cos(xx) + np.cos(yy))

    plot = Plot(xlim=klim, xlabel="$k_x$", ylim=klim, ylabel="$k_y$")
    plot.set_ticks(ticks, ticks)
    plot.set_ticklabels(tick_labels, tick_labels)
    # plot.ax.set_aspect("equal")
    levels = np.arange(-4, 4, 0.5)
    plot.ax.contourf(xx, yy, zz, levels=levels, cmap="viridis", vmax=4)

    loff = 0.02
    x = 0, np.pi-loff, np.pi-loff, 0
    y = 0, 0, np.pi, 0
    plot.plot(x, y, color="k")

    off = 0.1
    plot.text((-off, 0), r"$\Gamma$", ha="right")
    plot.text((np.pi + off, 0), r"$X$", ha="left")
    plot.text((np.pi + off, np.pi + off), r"$M$", ha="left")

    plot.ax.set_aspect("equal")
    plot.format_latex(textwidth=0.5, ratio=0.95)
    plot.save(SCRIPT, "tb2d_disp_contour.eps")
    # plot.show()


def calculate_dos2d():
    size = 128
    model = TbDevice.square((2, size))
    e = np.linspace(-6, 6, 1000)
    dos = model.lead.dos(e + eta, "b")
    ymax = int(np.ceil(np.max(dos) / 10) * 10)
    plot = Plot(xlim=(-6, 6), xlabel=r"$(E-\epsilon_0)/t$", ylim=(0, ymax), ylabel="$n(E)$")
    plot.grid()
    plot.plotfill(e, dos)
    plot.format_latex(textwidth=0.6)
    plot.save(SCRIPT, "tb_dos.eps")
    # plot.show()


# =========================================================================
# Transport theory
# =========================================================================


def plot_rda(omega=eta):
    n_iter = 40
    ham = np.array([[0]])
    t = np.array([[1]])
    eye = np.eye(ham.shape[0])
    alpha = t
    beta = np.conj(t).T
    h_l, h_b = ham.copy(), ham.copy()
    gf_surf = np.zeros(n_iter, "complex")
    gf_bulk = np.zeros(n_iter, "complex")
    for i in prange(n_iter):
        # calculate greens-functions
        gf_l = la.inv(omega * eye - h_l)
        gf_b = la.inv(omega * eye - h_b)

        # Recalculate hamiltonians
        h_l = h_l + alpha @ gf_b @ beta
        h_b = h_b + alpha @ gf_b @ beta + beta @ gf_b @ alpha
        gf_surf[i] = gf_l[0][0]
        gf_bulk[i] = gf_b[0][0]
        # Renormalize effective hopping
        alpha = alpha @ gf_b @ alpha
        beta = beta @ gf_b @ beta

    plot = Plot()
    plot.format_latex(textwidth=0.6)
    plot.set_labels(xlabel="Iteration")
    x = np.arange(n_iter)
    y = np.array(gf_surf)

    plot.plot(x[1:], np.imag(y)[1:], label=r"$ Im \mathcal{G}_{L/R}$")
    y = np.array(gf_bulk)
    plot.plot(x[1:], np.imag(y)[1:], label=r"$ Im \mathcal{G}_{B}$")
    plot.legend()
    plot.save(SCRIPT, "rda_it.eps", dpi=600)
    # plot.show()


def plot_surface_gf_1d():
    n = 500
    xmax = 4
    ymax = 1.1
    xlim = -xmax, xmax
    omegas = np.linspace(*xlim, n) + eta
    model = TbDevice.square((2, 1))
    lead = model.lead

    gf_s = np.zeros(n, "complex")
    gf_b = np.zeros(n, "complex")
    for i in prange(n):
        lead.calculate(omegas[i])
        gf_s[i] = lead.gf_r[0][0]
        gf_b[i] = lead.gf_b[0][0]

    plot = Plot(xlim=xlim, ylim=(-ymax, ymax), xlabel=r"$\omega$")
    plot.format_latex(textwidth=0.6, ratio=0.8)
    plot.plot(omegas.real, - gf_s.imag, label=r"$- |t| Im \mathcal{G}_{L/R}$")
    plot.fill(omegas.real, - gf_s.imag, alpha=0.25)
    plot.plot(omegas.real, + gf_s.real, color="k", lw=1, label=r"$+ |t| Re \mathcal{G}_{L/R}$")
    plot.grid()
    plot.legend()

    ymax = 2
    plot2 = Plot(xlim=xlim, ylim=(-ymax, ymax), xlabel=r"$\omega$")
    plot2.format_latex(textwidth=0.6, ratio=0.8)
    plot2.plot(omegas.real, - gf_b.imag, label=r"$- |t| Im \mathcal{G}_B$")
    plot2.fill(omegas.real, - gf_b.imag, alpha=0.25)
    plot2.plot(omegas.real, + gf_b.real, color="k", lw=1, label=r"$+ |t| Re \mathcal{G}_B$")
    plot2.grid()
    plot2.legend()

    plot.save(SCRIPT, "rda_gf_a.eps", dpi=600)
    plot2.save(SCRIPT, "rda_gf_b.eps", dpi=600)


def plot_surface_gf_2d():
    n = 1000
    xmax = 6
    ymax = 1.1
    xlim = -xmax, xmax
    omegas = np.linspace(*xlim, n) + eta
    model = TbDevice.square((2, 5))
    lead = model.lead

    gf_sc = np.zeros(n, "complex")
    gf_ss = np.zeros(n, "complex")
    gf_bc = np.zeros(n, "complex")
    gf_bs = np.zeros(n, "complex")
    idx = 2, 2
    for i in prange(n):
        lead.calculate(omegas[i])
        gf_sc[i] = lead.gf_r[idx]
        gf_ss[i] = lead.gf_r[0, 0]
        gf_bc[i] = lead.gf_b[idx]
        gf_bs[i] = lead.gf_b[0, 0]

    c = "C0"
    plot = Plot(xlim=xlim, ylim=(-ymax, ymax), xlabel=r"$\omega$")
    plot.format_latex(textwidth=0.6, ratio=0.8)
    plot.plot(omegas.real, - gf_sc.imag, color=c, label=r"$- |t| Im \mathcal{G}_{L/R}$")
    plot.fill(omegas.real, - gf_sc.imag, color=c, alpha=0.25)
    plot.plot(omegas.real, - gf_ss.imag, color=c, ls="--")
    # plot.fill(omegas.real, - gf_ss.imag, color=c, alpha=0.25)
    plot.plot(omegas.real, + gf_sc.real, color="k", lw=1, label=r"$+ |t| Re \mathcal{G}_{L/R}$")
    plot.plot(omegas.real, + gf_ss.real, color="k", lw=1, ls="--")
    plot.grid()
    plot.legend()

    ymax = 2
    plot2 = Plot(xlim=xlim, ylim=(0, ymax), xlabel=r"$\omega$")
    plot2.format_latex(textwidth=0.6, ratio=0.8)
    plot2.plot(omegas.real, - gf_bc.imag, color=c, label=r"$- |t| Im \mathcal{G}_{L/R}$")
    plot2.fill(omegas.real, - gf_bc.imag, color=c, alpha=0.25)
    plot2.plot(omegas.real, - gf_bs.imag, color=c, ls="--")
    # plot.fill(omegas.real, - gf_ss.imag, color=c, alpha=0.25)
    # plot2.plot(omegas.real, + gf_bc.real, color="k", lw=1, label=r"$+ |t| Re \mathcal{G}_{L/R}$")
    #plot2.plot(omegas.real, + gf_bs.real, color="k", lw=1, ls="--")
    plot2.grid()
    plot2.legend()

    plot.save(SCRIPT, "rda_gf2d_a.eps", dpi=600)
    plot2.save(SCRIPT, "rda_gf2d_b.eps", dpi=600)
    # plot.show()


def time_trans(model, rec):
    t0 = time.perf_counter()
    model.transmission(rec_thresh=rec)
    return time.perf_counter() - t0


def plot_rgf_speed(n_avrg=200):
    model = TbDevice.square((5, 5))
    l0, l1 = 10, 200
    lengths = np.arange(l0, l1+1, 10)
    n = lengths.shape[0]
    times_1 = np.zeros((n, n_avrg))
    times_2 = np.zeros((n, n_avrg))

    with Progress(total=2*n_avrg*n) as p:
        for i in range(n):
            model.reshape(lengths[i])
            for j in range(n_avrg):
                p.update()
                times_1[i, j] = time_trans(model, rec=1000000)
            for j in range(n_avrg):
                p.update()
                times_2[i, j] = time_trans(model, rec=1)

    plot = Plot()
    plot.set_labels(xlabel="$N$", ylabel="t[s]")
    plot.set_limits(xlim=[l0, l1])
    plot.plot(lengths, np.mean(times_1, axis=1), label="ED", color="black")
    plot.plot(lengths, np.mean(times_2, axis=1), label="RGF")
    plot.legend()
    # plot.show()
    plot.format_latex(textwidth=0.6)
    plot.save(SCRIPT, "rgf_speed.eps", dpi=DPI)


# =========================================================================
# Transport
# =========================================================================


def plot_1dchain_trans():
    elim = -2.5, 2.5
    omegas = np.linspace(*elim, 1000)
    plot = Plot(xlim=elim, ylim=(0, 1.09), xlabel=r"$E - \epsilon_0 \ [t]$", ylabel=r"$T(E)$")
    model = TbDevice.square((10, 1))
    trans = model.transmission_curve(omegas + eta)
    plot.plot(omegas, trans, label=r"RDA: $N=\infty$", color="k")
    omegas = np.linspace(*elim, 5000)

    for n in [5, 10]:
        model.reshape(n)
        model.load_lead(True)
        trans = model.transmission_curve(omegas + eta)
        plot.plot(omegas, trans, label=f"WB: $N={n}$", lw=1)
    plot.grid()
    plot.legend()
    plot.format_latex(textwidth=0.6, ratio=0.7)
    plot.save(SCRIPT, "tbtrans_1d.eps", dpi=DPI)
    #plot.show()


def plot_1dribbon_trans():
    elim = -4.5, 4.5
    omegas = np.linspace(*elim, 1000)
    n = 20
    m = 3
    model = TbDevice.square((5, m))
    trans_rda = model.transmission_curve(omegas + eta) / m
    model.reshape(n)
    model.load_lead(True)
    trans_wba = model.transmission_curve(omegas + eta) / m

    plot = Plot(xlim=elim, ylim=(0, 1.09), xlabel=r"$E - \epsilon_0 \ [t]$", ylabel=r"$T(E)/M$")
    plot.plot(omegas, trans_rda, label=r"RDA: $N=\infty$", color="k")
    plot.plot(omegas, trans_wba, label=f"WB: $N={n}$", lw=1)
    plot.grid()
    plot.legend()
    plot.format_latex(textwidth=0.6, ratio=0.7)

    m = 7
    model = TbDevice.square((5, m))
    trans_rda = model.transmission_curve(omegas + eta) / m
    model.reshape(n)
    model.load_lead(True)
    trans_wba = model.transmission_curve(omegas + eta) / m

    plot2 = Plot(xlim=elim, ylim=(0, 1.09), xlabel=r"$E - \epsilon_0 \ [t]$", ylabel=r"$T(E)/M$")
    plot2.plot(omegas, trans_rda, label=r"RDA: $N=\infty$", color="k")
    plot2.plot(omegas, trans_wba, label=f"WB: $N={n}$", lw=1)
    plot2.grid()
    plot2.legend()
    plot2.format_latex(textwidth=0.6, ratio=0.7)

    plot.save(SCRIPT, "tbtrans_2da.eps")
    plot2.save(SCRIPT, "tbtrans_2db.eps")


# =========================================================================
# Disorder
# =========================================================================


def plot_tcurve():
    n = 1000
    w_values = [0.5, 1]

    m = 1
    elim = -3, 3
    omegas = np.linspace(*elim, n) + eta
    model = TbDevice.square((10, m))
    plot = Plot(xlim=elim, ylim=(0, 1.09), xlabel=r"$E - \epsilon_0 \ [t]$", ylabel=r"$T(E)$")
    trans = model.transmission_curve(omegas) / m
    plot.plot(omegas.real, trans, color="k", lw=1)
    for w in w_values:
        model.set_disorder(w)
        trans = model.transmission_curve(omegas) / m
        plot.plot(omegas.real, trans, label=f"$w_\epsilon={w:.1f}$")
    plot.grid()
    plot.legend(loc="upper right")

    m = 3
    elim = -5, 5
    omegas = np.linspace(*elim, n) + eta
    model = TbDevice.square((10, m))
    plot2 = Plot(xlim=elim, ylim=(0, 1.09), xlabel=r"$E - \epsilon_0 \ [t]$", ylabel=r"$T(E) / M$")
    trans = model.transmission_curve(omegas) / m
    plot2.plot(omegas.real, trans, color="k", lw=1)
    for w in w_values:
        model.set_disorder(w)
        trans = model.transmission_curve(omegas) / m
        plot2.plot(omegas.real, trans, label=f"$w_\epsilon={w:.1f}$")
    plot2.grid()
    plot2.legend(loc="upper right")

    plot.format_latex(textwidth=0.6, ratio=0.7)
    plot2.format_latex(textwidth=0.6, ratio=0.7)

    plot.save(SCRIPT, "disordtrans_a.eps")
    plot2.save(SCRIPT, "disordtrans_b.eps")
    plot.show()


def plot_trans_hist():
    n = 100000
    model = TbDevice.square((200, 1))

    plot = Plot(xlabel=r"$\ln T$", ylabel=r"$P(\ln T)$")
    # plot.set_scales(xscale="log")
    plot.set_ticks(yticks=[])
    i = 0
    for w in [0.5, 1]:
        model.set_disorder(w)
        trans = model.mean_transmission(n=n, flatten=False)
        trans = np.log(trans)
        t_mean = np.mean(trans)

        col = f"C{i}"
        bins = np.linspace(min(trans), 0, 100)
        plot.histogram(trans, bins=bins, label=f"w={w:.1f}", color=col, alpha=0.5)
        plot.lines(x=t_mean, color=col)
        i += 1
    plot.legend()
    plot.format_latex(textwidth=0.6)
    plot.save(SCRIPT, "trans_hist.eps")
    plot.show()


def plot_transloss():
    model = TbDevice.square((2, 1))
    model.set_disorder(1)
    lengths = np.arange(2, 201, 2)

    trans = model.transmission_loss(lengths, n_avrg=1000)
    trans = np.mean(np.log(trans), axis=1)
    ll, llerr, fit_data = loc_length_fit(lengths, trans)
    print(ll, llerr)
    plot = Plot(xlim=(0, lengths[-1]), xlabel="$L [a]$", ylabel=r"$\ln T$")
    plot.plot(lengths, trans, label=r"$\langle \ln T \rangle$")
    plot.plot(*fit_data, ls="--", lw=1, label=r"$\ln T_{fit}$", color="k")

    plot.legend()
    plot.format_latex(textwidth=0.6, ratio=0.7)
    plot.save(SCRIPT, "disordtransloss.eps")
    # plot.show()


def plot_loclen_energy():
    model = TbDevice.square((2, 3))
    model.set_disorder(1)
    n = 100
    elim = -4, 4
    n_avrg = 200
    lengths = np.arange(20, 101, 5)

    energies = np.linspace(*elim, n)
    loclens, errs = np.zeros(n), np.zeros(n)
    with Progress(total=n * n_avrg * lengths.shape[0]) as p:
        for i in range(n):
            trans = model.transmission_loss(omega=energies[i] + eta, lengths=lengths, n_avrg=n_avrg, prog=p)
            trans = np.mean(np.log(trans), axis=1)
            loclens[i], errs[i] = loc_length(lengths, trans)

    plot = Plot(xlim=elim, xlabel=r"$E - \epsilon_0$ $[t]$", ylabel=r"$\xi_{loc}$")
    plot.ax.errorbar(energies, loclens, yerr=errs)
    plot.format_latex(textwidth=0.6, ratio=0.7)
    plot.save(SCRIPT, "disordtransenergy_b.eps")
    plot.show()


def dirarrow(plot, x, dx, y1, y2):
    dy = y2 - y1
    plot.ax.arrow(x, y1, dx, dy, shape='full', color="k",
                  lw=0, length_includes_head=False, head_width=.25)


def plot_scaling_beta():

    def beta(g, d):
        return (d-1) - (1+g) * np.log(1 + 1/g)

    xmax = 4
    g = np.geomspace(np.exp(-xmax), np.exp(xmax), 300)

    plot = Plot(xlim=(-xmax, xmax), xlabel=r"$\ln g$", ylim=(-3, 1.5), ylabel=r"$\beta$")
    plot.lines(x=0, y=0, color="0.3", lw=0.5)

    x = np.geomspace(np.exp(-1e-2), np.exp(xmax), 100)
    plot.plot(np.log(x), -1-1/(2*x), ls="--", lw=0.5, color="r")
    x = np.geomspace(np.exp(-xmax), np.exp(1e-2), 100)
    plot.plot(np.log(x), np.log(x) - x*(np.abs(np.log(x)) + 1), ls="--", lw=0.5, color="r")
    x, dx = -2, -0.00001
    for d in [1, 2, 3]:
        plot.plot(np.log(g), beta(g, d), color="k")
        plot.text((xmax-1, 1.6 - d), "$d=" + f"{4-d}" + "$")
    for d in [1, 2]:
        for x in [-3, -1, 1, 3]:
            dirarrow(plot, x, dx, beta(np.exp(x), d), beta(np.exp(x + dx), d))
    d = 3
    dirarrow(plot, -3, dx, beta(np.exp(-3), d), beta(np.exp(-3 + dx), d))
    for x in [-1, 1, 3]:
        dirarrow(plot, x, -dx, beta(np.exp(x), d), beta(np.exp(x - dx), d))

    plot.format_latex(textwidth=0.6)
    plot.save(SCRIPT, "scalingbeta.eps")
    plot.show()


# =========================================================================
# Anderson transition
# =========================================================================


def plot_p3_ham():
    n = 3
    b = p3_basis(soc=1)
    model = TbDevice.square((n, 1), basis=b)
    ham = model.hamiltonian(1)
    labels = list()
    for i in range(n):
        labels += b.labels()
    labels = labels + labels + labels
    plot = ham.show(cmap="Greys", show=False)
    plot.set_ticklabels(labels, labels)
    for i in range(n-1):
        r = i * 6
        c = (i + 1) * 6
        plot.frame([r, r + 6], [c, c + 6], color="C0", lw=1.5)
        plot.frame([c, c + 6], [r, r + 6], color="C0", lw=1.5)
    for i in range(n):
        r = i*6
        plot.frame([r, r+6], [r, r+6], color="r", lw=1.5)
    plot.format_latex()
    plot.save(SCRIPT, "p3ham.eps")
    plot.show()


def sort_keys(data):
    keys = data.keylist
    values = [data[k] for k in keys]
    key_vals = [data.key_value(k) for k in keys]
    idx = np.argsort(key_vals)
    data.clear()
    for i in idx:
        data.update({keys[i]: values[i]})
    data.save()


def get_data(data):
    w, ll, errs = list(), list(), list()
    for k in data:
        w.append(data.key_value(k))
        l, t = data.get_set(k, mean=False)
        t = np.mean(np.log(t), axis=1)
        lam, lam_err = loc_length(l, t)
        ll.append(lam)
        errs.append(lam_err)
    w = np.array(w)
    ll = np.array([ll, errs])
    return w, ll


def read_loclen_data(*txt):
    folder = Folder(DATA_DIR, "localization")
    data_list = list()
    for path in folder.find(*txt):
        data = LT_Data(path)
        sort_keys(data)
        h = data.info()["h"]
        w, ll = get_data(data)
        # Normalizing data
        ll /= h
        data_list.append((h, w, ll[0], ll[1]))
    return data_list


def plot_soc_transloss():
    plot = Plot(xlabel="$N$", ylabel=r"$\langle \ln T \rangle$")

    folder = Folder(DATA_DIR, "localization", "p3-basis")
    for path in folder.find("-h=1-"):
        data = LT_Data(path)
        k = "w=1"
        info = data.info()
        m, soc = info["h"], info["soc"]
        if soc in [0, 1, 2, 3]:
            lengths, trans = data.get_set(k)
            trans = np.mean(np.log(trans), axis=1)
            lam, lam_err = loc_length(lengths, trans)
            label = r"$\lambda_{soc}=" + f"{soc:.1f}" + "$"
            label += r", $\xi_{loc}\approx" + f"{lam:.0f}" + "$"
            plot.plot(lengths, trans, label=label)
        # plot.plot(*fit_data, ls="--", lw=1, label=r"$\ln T_{fit}$", color="k")

    plot.legend()
    plot.format_latex(textwidth=0.6, ratio=0.8)
    plot.save(SCRIPT, "at_trans1D.eps")
    # plot.show()


def plot_chainloclen():
    plot = Plot(xlabel=r"$w_{\epsilon}[t]$", ylabel=r"$\xi_{loc}$")
    plot.set_scales(yscale="log")
    plot.set_ticks(np.arange(0, 18, 2))
    xmax = 0
    folder = Folder(DATA_DIR, "localization", "p3-basis")
    for path in folder.find("-h=1-"):
        data = LT_Data(path)
        info = data.info()
        m, soc = info["h"], info["soc"]
        if soc in [0, 1, 2, 3]:
            label = r"$\lambda_{soc}=" + f"{soc:.1f}" +  "$"
            w, lldata = get_data(data)
            xmax = max(xmax, max(w))
            plot.ax.errorbar(w, lldata[0]/m, yerr=lldata[1]/m, label=label)
    # plot.set_limits(xlim=(0, xmax+2))
    plot.legend()
    plot.format_latex(textwidth=0.6, ratio=0.8)
    plot.save(SCRIPT, "at_loclen1D.eps")
    # plot.show()


def plot_p3banddos(t_pps=1., t_ppp=1., soc=1., h=64):
    n = 200
    size = 2
    d = np.array([1, 1, 1])
    b = p3_basis(eps_p=0, t_pps=t_pps, t_ppp=t_ppp, d=d, soc=soc)
    model = TbDevice.square((size, h), basis=b)

    band_sections = model.bands(POINTS)
    bands, ticks, ticklabels = build_bands(band_sections, NAMES)

    bmin, bmax = np.nanmin(bands), np.nanmax(bands)
    bandwidth = bmax - bmin
    elim = bmin - 0.1*bandwidth, bmax + 0.1*bandwidth
    omegas = np.linspace(*elim, n) + eta
    dos = model.lead.dos(omegas, "b")

    energy = omegas.real
    plot = Plot.banddos(elim=elim, ratio=2)
    plot.switch_axis(0)
    plot.grid(axis="x")
    plot.set_limits((ticks[0], ticks[1]))
    plot.set_ticks(ticks)
    plot.set_ticklabels(ticklabels)
    plot.lines(x=ticks[1:-1], y=0, lw=0.5, color="0.5")
    plot.plot(bands)

    plot.switch_axis(1)
    plot.set_limits((0, max(dos)*1.2))
    plot.set_ticklabels(yticks=[])
    plot.lines(y=0, lw=0.5, color="0.5")
    plot.plotfill(dos, energy, color="k", lw=1, alpha=0.15)

    plot.format_latex(textwidth=0.8)
    # plot.save(SCRIPT, f"p3_banddos_soc_{soc}.eps")
    return plot


def plot_ribbond_loclen(soc=1):
    folder = Folder(DATA_DIR, "localization", "p3-basis", f"soc={soc}")

    plot = Plot(xlabel=r"$w_{\epsilon}[t]$", ylabel=r"$\Lambda = \xi_{loc} / M$")
    plot.set_scales(yscale="log")
    plot.set_ticks(np.arange(0, 18, 2))
    datas = [LT_Data(path) for path in folder.files]
    heights = [d.info()["h"] for d in datas]
    idx = np.argsort(heights)
    for i in idx:
        data = datas[i]
        m = data.info()["h"]
        label = r"$M=" + f"{m:.0f}" + "$"
        w, lldata = get_data(data)
        plot.ax.errorbar(w, lldata[0]/m, yerr=lldata[1]/m, label=label)
    plot.legend()
    plot.format_latex(textwidth=0.6, ratio=0.8)
    plot.save(SCRIPT, f"at_soc_{soc}.eps")
    plot.show()


def main():
    Plot.enable_latex()
    # plot_chain_dos()
    # plot_chain(textwidth=0.6, ratio=0.8)
    # plot_bandstruct_2d()
    # calculate_dos2d()
    # plot_rda()
    # plot_chain()
    # plot_surface_gf_1d()
    # plot_surface_gf_2d()
    # plot_rgf_speed()
    # plot_1dchain_trans()
    # plot_1dribbon_trans()
    # plot_tcurve()
    # plot_trans_hist()
    # plot_transloss()
    # plot_loclen_energy()
    # plot_scaling_beta()
    # plot_p3_ham()
    # plot_p3banddos(soc=5)
    # plot_soc_transloss()
    # plot_chainloclen()
    # plot_ribbond_loclen(soc=2)


if __name__ == "__main__":
    main()
