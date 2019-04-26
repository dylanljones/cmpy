# -*- coding: utf-8 -*-
"""
Created on 5 Dec 2018
@author: Dylan Jones

project: tightbinding
version: 1.0

Plotting utilities
==================

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors


class Plot:

    def __init__(self, xlim=None, xlabel=None, ylim=None, ylabel=None, title=None, proj=None):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection=proj)

        self.set_limits(xlim, ylim)
        self.set_labels(xlabel, ylabel)
        self.set_title(title)

    @property
    def xlim(self):
        return self.ax.get_xlim()

    @property
    def ylim(self):
        return self.ax.get_ylim()

    def set_scales(self, xscale=None, yscale=None, zscale=None):
        if xscale is not None:
            self.ax.set_xscale(xscale)
        if yscale is not None:
            self.ax.set_yscale(yscale)
        if zscale is not None:
            self.ax.set_zscale(zscale)

    def set_title(self, txt=None):
        if txt:
            self.ax.set_title(txt)

    def set_limits(self, xlim=None, ylim=None, zlim=None):
        if xlim is not None:
            self.ax.set_xlim(*xlim)
        if ylim is not None:
            self.ax.set_ylim(*ylim)
        if zlim is not None:
            self.ax.set_zlim(*zlim)

    def set_labels(self, xlabel=None, ylabel=None, zlabel=None):
        if xlabel is not None:
            self.ax.set_xlabel(xlabel)
        if ylabel is not None:
            self.ax.set_ylabel(ylabel)
        if zlabel is not None:
            self.ax.set_zlabel(zlabel)

    def set_ticklabels(self, xticks=None, yticks=None, zticks=None):
        if xticks is not None:
            self.ax.set_xticklabels(xticks)
        if yticks is not None:
            self.ax.set_yticklabels(yticks)
        if zticks is not None:
            self.ax.set_zticklabels(zticks)

    def lines(self, x=None, y=None, *args, **kwargs):
        if x is not None:
            if not hasattr(x, "__len__"):
                x = [x]
            for _x in x:
                self.ax.axhline(_x, *args, **kwargs)
        if y is not None:
            if not hasattr(y, "__len__"):
                y = [y]
            for _y in y:
                self.ax.axvline(_y, *args, **kwargs)

    def plot(self, *args, **kwargs):
        return self.ax.plot(*args, **kwargs)[0]

    def tight(self, *args, **kwargs):
        self.fig.tight_layout(*args, **kwargs)

    def legend(self):
        self.ax.legend()

    @staticmethod
    def draw(sleep=1e-10):
        plt.draw()
        plt.pause(sleep)

    def show(self, tight=True):
        if tight:
            self.tight()
        plt.show()


class MatrixPlot(Plot):

    def __init__(self, cmap="viridis"):
        super().__init__()
        self.array = None
        self.shape = 0, 0
        self.cmap = cm.get_cmap(cmap)
        self.norm = None

    def load(self, array):
        self.array = np.abs(array)
        self.shape = self.array.shape
        self.norm = colors.Normalize(vmin=np.min(self.array), vmax=np.max(self.array))
        self._draw()

    def line(self, row=None, col=None, color="white", lw=1, **kwargs):
        if col is not None:
            self.ax.axvline(x=self._get_x(col), color=color, lw=lw, **kwargs)
        if row is not None:
            self.ax.axhline(y=self._get_y(row), color=color, lw=lw, **kwargs)

    def frame(self, rows, cols, color="white", lw=1, **kwargs):
        x0, y0 = self._get_xy(rows[0], cols[0])
        x1, y1 = self._get_xy(rows[1], cols[1])
        self.ax.plot([x0, x0], [y0, y1], color=color, lw=lw, **kwargs)
        self.ax.plot([x1, x1], [y0, y1], color=color, lw=lw, **kwargs)
        self.ax.plot([x0, x1], [y0, y0], color=color, lw=lw, **kwargs)
        self.ax.plot([x0, x1], [y1, y1], color=color, lw=lw, **kwargs)

    def text(self, row, col, txt, offset=None, va="center", ha="center", **kwargs):
        (x_off, y_off) = (0, 0) if offset is None else offset
        x0, y0 = self._get_xy(row, col)
        x1, y1 = self._get_xy(row+1, col+1)
        x = x0 + 0.5 * abs(x1-x0) + x_off
        y = y0 - 0.5 * abs(y1-y0) + y_off
        self.ax.text(x, y, s=txt, va=va, ha=ha, **kwargs)

    def _draw(self, numbering_max=25):
        n, m = self.shape
        if max(n, m) < 50:
            for r in range(n):
                for c in range(m):
                    color = self.cmap(self.norm(self.array[r, c]))
                    self._draw_element(r, c, color)
            self._draw_grid()
        else:
            xx, yy = np.meshgrid(range(n+1), range(m+1))
            self.ax.pcolormesh(xx, -yy, self.array.T)
        numbering = not np.any([x > numbering_max for x in self.shape])
        self._set_ticks(numbering)
        self.ax.set_xlim(0, self.shape[1])
        self.ax.set_ylim(-self.shape[0], 0)
        self.ax.set_aspect("equal", "box")
        self.fig.tight_layout()

    def _draw_element(self, row, col, color, **kwargs):
        x0, y0 = self._get_xy(row, col)
        x1, y1 = self._get_xy(row+1, col+1)
        self.ax.fill_between([x0, x1], y0, y1, color=color, **kwargs)

    def _draw_grid(self, lw=0.5, color="0.3", **kwargs):
        for x in range(1, self.shape[1]):
            self.ax.axvline(x, lw=lw, color=color, **kwargs)
        for y in range(-self.shape[0]+1, 0):
            self.ax.axhline(y, lw=lw, color=color, **kwargs)

    def _set_ticks(self, numbering=False):
        rows, cols = self.shape
        if numbering:
            self.ax.set_xticks(np.arange(cols+1) + 0.5)
            self.ax.set_xticklabels([str(i+1) for i in range(cols+1)])
            self.ax.set_yticks(np.arange(-rows, 0) + 0.5)
            self.ax.set_yticklabels([str(i+1) for i in reversed(range(rows))])
        else:
            self.ax.set_xticks([0.5, cols-0.5])
            self.ax.set_xticklabels(["1", str(cols)])
            self.ax.set_yticks([-rows+0.5, -0.5])
            self.ax.set_yticklabels([str(rows), "1"])

    def set_ticklabels(self, labels):
        self.ax.set_xticklabels(labels, rotation = 45, ha="right")
        self.ax.set_yticklabels(labels[::-1])

    @staticmethod
    def _get_x(col):
        return col

    @staticmethod
    def _get_y(row):
        return -row

    def _get_xy(self, row, col):
        return self._get_x(col), self._get_y(row)

    @staticmethod
    def show():
        plt.show()


class LatticePlot(Plot):

    def __init__(self, dim=2):
        proj = '3d' if dim == 3 else None
        super().__init__(proj=proj)

        self._dim = dim
        self._colors = dict()
        self._limits = np.zeros((dim, 2))

    def draw_site(self, atype, pos):
        if atype in self._colors:
            col = self._colors[atype]
        else:
            col = f"C{len(self._colors)}"
            self._colors.update({atype: col})
        self.update_limits(pos)
        self.ax.scatter(*pos, color=col, zorder=1, s=100)

    def update_limits(self, pos):
        for i in range(self._dim):
            if pos[i] < self._limits[i, 0]:
                self._limits[i, 0] = pos[i]
            if pos[i] > self._limits[i, 1]:
                self._limits[i, 1] = pos[i]

    def set_ticks(self):
        self.ax.set_xticks(self._limits[0])
        self.ax.set_yticks(self._limits[1])
        if self._dim == 3:
            self.ax.set_zticks(self._limits[2])

    def rescale(self, offset=1):
        limits = list()
        for i in range(self._dim):
            ax_lim = self._limits[i, 0]-offset, self._limits[i, 1]+offset
            limits.append(ax_lim)
        self.set_limits(*limits)
        self.set_ticks()

    def draw_line(self, points, color="black", **kwargs):
        points = np.asarray(points)
        self.ax.plot(*points.T, color=color, zorder=0, **kwargs)


def band_dos_plot(omegas, dos, band_sections, point_names, show=True):
    omegas = omegas.real
    ylim = min(omegas), max(omegas)

    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})

    # Formatting
    ax1.set_ylim(*ylim)
    ax1.axhline(y=0, lw=0.5, color="r")
    ax2.set_ylim(*ylim)
    ax2.axhline(y=0, lw=0.5, color="r")

    # Plot the band-structure
    point_names = list(point_names)
    point_names.append(point_names[0])
    n_sections = len(band_sections)
    x0, ticks = 0, [0]
    for i in range(n_sections):
        section = band_sections[i]
        x = x0 + np.arange(section.shape[0])
        ax1.plot(x, section, color="black")
        x0 = max(x) + 1
        ticks.append(x0)
    ax1.set_xticks(ticks)
    ax1.set_ylabel("Energy")
    ax1.set_xlabel(r"$\vec{k}$")
    ax1.set_xlim(0, ticks[-1])

    ax1.set_xticklabels(point_names)
    for x_p in ticks[1:-1]:
        ax1.axvline(x=x_p, lw=0.5, color="0.5")

    # Plot the dos
    ax2.plot(dos, omegas, color="k")
    ax2.fill_between(dos, 10, omegas, color="0.5", alpha=0.5)
    ax2.set_xticks([max(dos)*1.1])
    ax2.set_xticklabels([""])
    ax2.set_xlim(0, max(dos)*1.1)
    ax2.set_xlabel("DOS")
    ax2.set_yticks([])
    ax2.set_yticklabels([])

    fig.subplots_adjust(wspace=0, hspace=0)
    # fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_transmission(omegas, *transmission, show=True):
    omegas = omegas.real
    t_lim = max([max(t) for t in transmission]) + 0.05

    fig, ax = plt.subplots()
    ax.set_title("Transmission")
    ax.set_xlim(min(omegas), max(omegas))
    ax.set_xlabel(r"$\omega$")
    ax.set_ylim(-0.05, t_lim)
    ax.axhline(y=0, lw=0.5, color="0.5")
    ax.set_ylabel(r"$T(\omega)$")

    for trans in transmission:
        ax.plot(omegas, trans)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_transmission_loss(lengths, trans, norm=1, mode="lin", show=True):
    fig, ax = plt.subplots()

    if mode == "exp":
        ax.set_yscale("log")
        ylabel = r"$T/T_0$"
    else:
        trans = np.log10(trans)
        ylabel = r"$\log(T/T_0)$"

    ax.plot(lengths, trans)

    ax.set_xlim(0, lengths[-1] + 10)
    ax.set_xlabel("N")
    ax.set_ylabel(ylabel)
    if show:
        plt.show()
    return fig, ax


def plot_dos(omegas, *dos, show=True):
    omegas = omegas.real

    fig, ax = plt.subplots()
    ax.set_title("Density of states")
    for dos_curve in dos:
        ax.plot(omegas, dos_curve)
    ax.axhline(y=0, lw=0.5, color="0.5")
    ax.set_xlim(min(omegas), max(omegas))
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho(\omega)$")

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_bands(band_sections, point_names, show=True):
    point_names = list(point_names)
    point_names.append(point_names[0])

    fig, ax = plt.subplots()
    ax.set_title("Band structure")
    n_sections = len(band_sections)
    x0, ticks = 0, [0]
    for i in range(n_sections):
        section = band_sections[i]
        x = x0 + np.arange(section.shape[0])
        ax.plot(x, section, color="black")
        x0 = max(x) + 1
        ticks.append(x0)

    ax.set_xlabel(r"$k$")
    ax.set_xlim(0, ticks[-1])
    ax.set_xticks(ticks)

    ax.set_xticklabels(point_names)
    ax.axhline(y=0, lw=0.5, color="r")
    for x_p in ticks[1:-1]:
        ax.axvline(x=x_p, lw=0.5, color="0.5")
    ax.set_ylabel(r"$E(k)$")

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax
