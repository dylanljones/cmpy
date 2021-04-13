# coding: utf-8
#
# This code is part of cmpy.
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Matplotlib-wrapper and plotting tools."""

import matplotlib.pyplot as plt


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

    def set_figsize(self, width, height, dpi=None):
        self.fig.set_size_inches(width, height)
        if dpi is not None:
            self.fig.set_dpi(dpi)

    def tight_layout(self):
        self.fig.tight_layout()

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
