# coding: utf-8
#
# This code is part of cmpy.
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Matplotlib-wrapper and plotting tools."""

import os
import matplotlib
import matplotlib.pyplot as plt
import colorcet as cc


def set_text_params(size=None, family=None):
    if size is not None:
        matplotlib.rcParams['font.size'] = size
    if family is not None:
        matplotlib.rcParams['font.family'] = family


def set_prop_cycle(**kwargs):
    cycler = matplotlib.cycler(**kwargs)
    matplotlib.rcParams['axes.prop_cycle'] = cycler


def use_latex(*packages, font_size=None):
    # LaTeX setup
    matplotlib.rc('text', usetex=True)

    # Set LaTeX-preamble
    if packages:
        key = 'text.latex.preamble'
        preamble = matplotlib.rcParams.get(key, "")
        new = "\n".join([r"\usepackage{" + p + "}" for p in packages])
        matplotlib.rcParams[key] = preamble + new

    if font_size is not None:
        matplotlib.rcParams['font.size'] = font_size


def use_colorcet(cmap=cc.glasbey_category10, n=10):
    set_prop_cycle(color=cmap[:n])


def get_figure_width(columns=1.0, column_width=3.50394):
    """Figure widths defined by Nature journal."""
    if columns == 1.0:
        return column_width
    elif columns == 2.0:
        return 7.204724
    else:
        return columns * column_width


def set_column_figsize(fig, columns=1.0, ratio=3/4, dpi=300):
    width = get_figure_width(columns)
    height = width * ratio
    fig.set_size_inches(width, height)
    if dpi is not None:
        fig.set_dpi(dpi)


def save_figure(fig, *relpaths, dpi=600, frmt=None, rasterized=True):
    print(f"Saving...", end="", flush=True)
    if rasterized:
        for ax in fig.get_axes():
            ax.set_rasterized(True)
    file = os.path.join(*relpaths)
    if (frmt is not None) and (not file.endswith(frmt)):
        filename, _ = os.path.splitext(file)
        file = filename + "." + frmt
    fig.savefig(file, dpi=dpi, format=frmt)
    print(f"\rFigure saved: {os.path.split(file)[1]}")
    return file


class Plot:
    """Matplotlib.pyplot.Axes wrapper which also supports some methods of the figure.

    Parameters
    ----------
    ax : plt.Axes, optional
        Optional existing Axes instance. If ``None`` a new subplot is created.
    """

    def __init__(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        self.fig, self.ax = fig, ax

    def __getattr__(self, item):
        return getattr(self.ax, item)

    def set_equal_aspect(self):
        self.ax.set_aspect('equal')

    def grid(self, b=None, which="major", axis="both", below=True, **kwargs):
        self.ax.set_axisbelow(below)
        self.ax.grid(b, which, axis, **kwargs)

    # =========================================================================

    def set_column_figsize(self, columns=1.0, ratio=3/4, dpi=300):
        width = get_figure_width(columns)
        height = width * ratio
        self.set_figsize(width, height, dpi)

    def set_figsize(self, width, height, dpi=None):
        self.fig.set_size_inches(width, height)
        if dpi is not None:
            self.fig.set_dpi(dpi)

    def tight_layout(self):
        self.fig.tight_layout()

    def save(self, *relpaths, dpi=600, frmt=None, rasterized=True):
        return save_figure(self.fig, *relpaths, dpi=dpi, frmt=frmt, rasterized=rasterized)

    def show(self, tight=True, block=True):
        if tight:
            self.tight_layout()
        plt.show(block=block)

    @staticmethod
    def pause(interval=0.):
        plt.pause(interval)

    def draw(self, pause=1e-10):
        self.fig.canvas.flush_events()
        plt.show(block=False)
        plt.pause(pause)
