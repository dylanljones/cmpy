# -*- coding: utf-8 -*-
"""
Created on  06 2019
author: dylan

project: sciutils
version: 1.0
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from .misc import OptionError

# Golden ratio as standard ratio for plot-figures
GOLDEN_RATIO = (np.sqrt(5) - 1.0) / 2.0
# Get the width from LaTeX using \showthe\columnwidth
# and set value of TEXTWIDTH to it
TEXTWIDTH = 455.2


class RcParams:

    @staticmethod
    def use_latex():
        plt.rcParams.update({'text.usetex': True})

    @staticmethod
    def set_latex_preamble(preamble):
        plt.rcParams.update({'text.latex.preamble': preamble})

    @staticmethod
    def set_fontsize(x):
        plt.rcParams.update({"font.size": x})

    @staticmethod
    def set_label_size(x):
        plt.rcParams.update({"axes.labelsize": x})

    @staticmethod
    def set_legend_fontsize(x):
        plt.rcParams.update({"legend.fontsize": x})

    @staticmethod
    def set_ticklabel_fontsize(x):
        values = {"xtick.labelsize": x,
                  "ytick.labelsize": x}
        plt.rcParams.update(values)

    def __str__(self):
        string = ""
        for key, val in plt.rcParams:
            string += f"{key:20} {val}\n"
        return string


def pts_to_inch(pts):
    return pts * (1. / 72.27)


def get_figsize(width=None, height=None, ratio=None):
    # Width and height
    if (width is not None) and (height is not None):
        width = pts_to_inch(width)
        height = pts_to_inch(height)
    else:
        if ratio is None:
            ratio = GOLDEN_RATIO
        # Width and ratio
        if width is not None:
            width = pts_to_inch(width)
            height = width * ratio
        # height and ratio
        elif height is not None:
            height = pts_to_inch(height)
            width = height / ratio
    return width, height


def get_latex_figsize(textwidth=0.8, ratio=None):
    width = textwidth * TEXTWIDTH
    return get_figsize(width=width, ratio=ratio)


def latex_format(font=None, fontsize=11, labelsize=None, legendsize=None, ticksize=8):
    RcParams.use_latex()
    if font:
        preamble = [r"\usepackage{" + font + "}"]
        RcParams.set_latex_preamble(preamble)

    labelsize = fontsize if labelsize is None else labelsize
    legendsize = fontsize if legendsize is None else legendsize
    ticksize = fontsize if ticksize is None else ticksize

    RcParams.set_fontsize(fontsize)
    RcParams.set_label_size(labelsize)
    RcParams.set_legend_fontsize(legendsize)
    RcParams.set_ticklabel_fontsize(ticksize)


class Plot:

    TEXTWIDTH = 455.2

    def __init__(self, xlim=None, xlabel=None, ylim=None, ylabel=None, title=None, proj=None, create=True):
        self.fig = plt.figure()
        self._axs, self.ax_idx = list(), 0
        if create:
            if proj == "3d":
                self._dim = 3
                self._axs.append(Axes3D(self.fig))
            else:
                self._dim = 2
                self.add_subplot()
            # self.ax = self.fig.add_subplot(111) if proj is None else Axes3D(self.fig)
            self.set_limits(xlim, ylim)
            self.set_labels(xlabel, ylabel)
            self.set_title(title)

    @classmethod
    def quickplot(cls, x, y, show=True):
        self = cls()
        self.plot(x, y)
        if show:
            self.show()
        return self

    # =========================================================================
    # Figure formatting
    # =========================================================================

    @staticmethod
    def enable_latex():
        RcParams.use_latex()

    @property
    def dpi(self):
        return self.fig.dpi

    @property
    def size(self):
        return self.fig.get_size_inches() * self.dpi

    def set_figsize(self, width=None, height=None, ratio=None):
        width, height = get_figsize(width, height, ratio)
        self.fig.set_size_inches(width, height)

    def set_latex_figsize(self, textwidth, ratio=None):
        width, height = get_latex_figsize(textwidth, ratio)
        self.fig.set_size_inches(width, height)

    def latex_plot(self, width=0.8, ratio=None, font=None, fontsize=11, labelsize=None, legendsize=None, ticksize=8):
        latex_format(font, fontsize, labelsize, legendsize, ticksize)
        self.set_latex_figsize(width, ratio)

    # =========================================================================
    # Axis formatting
    # =========================================================================

    @property
    def axs(self):
        return self._axs

    @property
    def ax(self):
        return self._axs[self.ax_idx]

    @property
    def xaxis(self):
        return self.ax.xaxis

    @property
    def yaxis(self):
        return self.ax.yaxis

    def switch_axis(self, idx):
        self.ax_idx = idx

    def get_ax(self, idx):
        self.switch_axis(idx)
        return self.ax

    def _add_ax(self, ax):
        idx = len(self._axs)
        self._axs.append(ax)
        return self.get_ax(idx)

    def add_subplot(self, num=111, *args, **kwargs):
        ax = self.fig.add_subplot(num, *args, **kwargs)
        return self._add_ax(ax)

    def add_xax(self):
        ax = self.ax.twiny()
        return self._add_ax(ax)

    def add_yax(self):
        ax = self.ax.twinx()
        return self._add_ax(ax)

    @property
    def lines(self):
        return self.ax.lines

    def get_data(self, idx=0):
        return self.ax.lines[idx].get_data()

    def get_data_point(self, x_rel, idx=0):
        x_data, y_data = self.get_data(idx)

        x0, x1 = self.get_datalimit(delta=0, axis=0)
        x = x0 + x_rel * abs(x1 - x0)

        idx = (np.abs(x_data - x)).argmin()
        return np.array([x_data[idx], y_data[idx]])

    def set_data(self, x, y, idx=0):
        self.ax.lines[idx].set_data(x, y)

    def set_ydata(self, y, idx=0):
        self.ax.lines[idx].set_ydata(y)

    @property
    def xlim(self):
        return self.ax.get_xlim()

    @property
    def ylim(self):
        return self.ax.get_ylim()

    @property
    def zlim(self):
        return self.ax.get_zlim()

    def set_title(self, txt=None):
        if txt:
            self.ax.set_title(txt)

    def set_equal_aspect(self):
        self.ax.set_aspect("equal", "box")

    def set_scales(self, xscale=None, yscale=None, zscale=None):
        if xscale is not None:
            self.ax.set_xscale(xscale)
        if yscale is not None:
            self.ax.set_yscale(yscale)
        if zscale is not None:
            self.ax.set_zscale(zscale)

    def set_limits(self, xlim=None, ylim=None, zlim=None):
        if xlim is not None:
            if not hasattr(xlim, "__len__"):
                xlim = self.get_datalimit(xlim, axis=0)
            self.ax.set_xlim(*xlim)
        if ylim is not None:
            if not hasattr(ylim, "__len__"):
                ylim = self.get_datalimit(ylim, axis=1)
            self.ax.set_ylim(*ylim)
        if zlim is not None:
            self.ax.set_zlim(*zlim)

    def set_margins(self, *args, **kwargs):
        self.ax.margins(*args, **kwargs)

    def get_datalimit(self, delta=0.1, axis=1):
        lines = self.ax.lines
        maxval = 1e100
        argmin, argmax = maxval, -maxval
        for line in lines:
            data = line.get_data()[axis]
            argmin = min(np.min(data), argmin)
            argmax = max(np.max(data), argmax)
        offset = delta * abs(argmax - argmin)
        return argmin - offset, argmax + offset

    def get_datarange(self, axis=0):
        limits = self.get_datalimit(delta=0., axis=axis)
        return abs(limits[1] - limits[0])

    def set_labels(self, xlabel=None, ylabel=None, zlabel=None):
        if xlabel is not None:
            self.ax.set_xlabel(xlabel)
        if ylabel is not None:
            self.ax.set_ylabel(ylabel)
        if zlabel is not None:
            self.ax.set_zlabel(zlabel)

    def set_ticks(self, xticks=None, yticks=None, zticks=None, minor=False):
        if xticks is not None:
            self.ax.set_xticks(xticks, minor)
        if yticks is not None:
            self.ax.set_yticks(yticks, minor)
        if zticks is not None:
            self.ax.set_zticks(zticks, minor)

    def set_ticklabels(self, xticks=None, yticks=None, zticks=None):
        if xticks is not None:
            self.ax.set_xticklabels(xticks)
        if yticks is not None:
            self.ax.set_yticklabels(yticks)
        if zticks is not None:
            self.ax.set_zticklabels(zticks)

    def color_axis(self, color, axis="y"):
        self.ax.tick_params(axis=axis, labelcolor=color)
        if axis == "x":
            self.xaxis.label.set_color(color)
        if axis == "y":
            self.yaxis.label.set_color(color)

    # =========================================================================
    # Drawing
    # =========================================================================

    def draw_lines(self, x=None, y=None, *args, **kwargs):
        if x is not None:
            if not hasattr(x, "__len__"):
                x = [x]
            for _x in x:
                self.ax.axvline(_x, *args, **kwargs)
        if y is not None:
            if not hasattr(y, "__len__"):
                y = [y]
            for _y in y:
                self.ax.axhline(_y, *args, **kwargs)

    def fill(self, x, y1, y2=0, alpha=0.25, *args, **kwargs):
        self.ax.fill_between(x, y1, y2, alpha=alpha, *args, **kwargs)

    def text(self, pos, text, va="center", ha="center"):
        self.ax.text(*pos, text, va=va, ha=ha)

    def annotate_data(self, string, x_rel=0.5, idx=0, offset=0.5, anchor="above"):
        point = self.get_data_point(x_rel, idx)
        if anchor == "above":
            offset = np.array([0, offset * self.get_datarange(1)])
            self.text(point + offset, string, va="bottom")
        elif anchor == "below":
            offset = np.array([0, offset * self.get_datarange(1)])
            self.text(point - offset, string, va="top")
        elif anchor == "right":
            offset = np.array([offset * self.get_datarange(0), 0])
            self.text(point + offset, string, ha="left")
        elif anchor == "left":
            offset = np.array([offset * self.get_datarange(0), 0])
            self.text(point - offset, string, ha="right")

    # =========================================================================
    # Plotting
    # =========================================================================

    def plot(self, *args, **kwargs):
        return self.ax.plot(*args, **kwargs)[0]

    def plotfill(self, x, y, color="C0", alpha=0.25, **kwargs):
        self.fill(x, y, color=color, alpha=alpha)
        return self.ax.plot(x, y, color=color, **kwargs)[0]

    def errorplot(self, x, y, yerr, alpha=0.4, *args, **kwargs):
        line = self.plot(x, y, *args, **kwargs)
        if not hasattr(yerr, "__len__"):
            yerr = np.array([yerr, yerr])
        else:
            if len(yerr) != 2:
                yerr = np.asarray([yerr, yerr])
            else:
                yerr = np.asarray(yerr)
        self.fill(x, y - yerr[0], y + yerr[1], color=line.get_color(), alpha=alpha)

    def scatter(self, x, y, s=None, *args, **kwargs):
        s = None if s is None else s**2
        return self.ax.scatter(x, y, s=s, *args, **kwargs)

    def contour(self, *args, **kwargs):
        return self.ax.contour(*args, **kwargs)

    def colormesh(self, *args, **kwargs):
        return self.ax.pcolormesh(*args, **kwargs)

    def histogram(self, data, bins=None, **kwargs):
        self.ax.hist(data, bins=bins, **kwargs)

    # =========================================================================
    # Other
    # =========================================================================

    def tight(self, *args, **kwargs):
        self.fig.tight_layout(*args, **kwargs)

    def legend(self, *args, **kwargs):
        self.ax.legend(*args, **kwargs)

    def grid(self, below_axis=True, **kwargs):
        self.ax.set_axisbelow(below_axis)
        self.ax.grid(**kwargs)

    def colorbar(self, im, *args, orientation="vertical", **kwargs):
        divider = make_axes_locatable(self.ax)
        if orientation == "vertical":
            cax = divider.append_axes("right", size="5%", pad=0.05)
        elif orientation == "horizontal":
            cax = divider.append_axes("bottom", size="5%", pad=0.6)
        else:
            raise OptionError(orientation, ["vertical", "horizontal"], name="orientation")
        return self.fig.colorbar(im, ax=self.ax, cax=cax, orientation=orientation, *args, **kwargs)

    @staticmethod
    def draw(sleep=1e-10):
        plt.draw()
        plt.pause(sleep)

    def show(self, tight=True):
        if tight:
            self.tight()
        plt.show()

    def save(self, *relpaths, dpi=600, rasterized=True):
        if rasterized:
            for ax in self._axs:
                ax.set_rasterized(True)
        file = os.path.join(*relpaths)
        self.fig.savefig(file, dpi=dpi)
        print(f"Figure {file} saved")

    def set_scalar_ticks(self, x=False, y=False, z=False):
        if x:
            self.ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        if y:
            self.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        if z:
            self.ax.zaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))


class MatrixPlot(Plot):

    def __init__(self, cmap="Greys", norm_offset=0.):
        super().__init__()
        self.array = None
        self.cmap = cm.get_cmap(cmap)
        self.norm = None
        self.noffset = norm_offset
        self.im = None

        self.set_equal_aspect()
        self.ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        self.ax.invert_yaxis()

    @property
    def shape(self):
        return self.array.shape

    def load(self, array):
        array = np.abs(array)
        self.array = array

        nlim = np.min(array), np.max(array)
        off = self.noffset * abs(nlim[1] - nlim[0])
        self.norm = colors.Normalize(vmin=nlim[0] - off, vmax=nlim[1] + off)

        xx, yy = np.meshgrid(range(self.shape[0] + 1), range(self.shape[1] + 1))
        self.im = self.colormesh(xx + 0.5, yy + 0.5, array, norm=self.norm, cmap=self.cmap)

    def show_values(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                x = f"{self.array[i, j]:.1f}"
                self.set_text(i, j, x)

    @staticmethod
    def get_position(i, j):
        return np.array([i+0.5, j+0.5])

    def show_colorbar(self):
        self.fig.colorbar(self.im, ax=self.ax)

    def set_text(self, i, j, string, **kwargs):
        center = self.get_position(i, j) + np.array((0.5, 0.5))
        self.text(center, string, **kwargs)

    def draw_grid(self, color="black"):
        self.grid(below_axis=False, which="minor", color=color)

    def draw_segment(self, points, cycle=False, **kwargs):
        points = list(points)
        if cycle:
            points.append(points[0])
        n = len(points)
        for i in range(n-1):
            p1, p2 = points[i:i+2]
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)

    def frame(self, pos, size, color="r", lw=1, **kwargs):
        x0, y0 = self.get_position(*pos)
        x1, y1 = x0 + size[0], y0 + size[1]
        points = (x0, y0), (x0, y1), (x1, y1), (x1, y0)
        self.draw_segment(points, cycle=True, color=color, lw=lw, **kwargs)

    def set_basislabels(self, xlabels=None, ylabels=None):
        if xlabels is not None:
            xlabels = [""] + xlabels
            self.ax.set_xticklabels(xlabels, rotation = 45, ha="right")
        if ylabels is not None:
            ylabels = [""] + ylabels
            self.ax.set_yticklabels(ylabels)
