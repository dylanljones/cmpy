# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2021, Dylan Jones

import os
import logging
import matplotlib.pyplot as plt

# =========================================================================
# LOGGING
# =========================================================================

# Configure package logger
logger = logging.getLogger("cmpy")

_stream_handler = logging.StreamHandler()
_stream_handler.setLevel(logging.DEBUG)

# _frmt_str = "[%(asctime)s] %(levelname)-8s - %(name)s - %(funcName)s - %(message)s"
_frmt_str = "[%(asctime)s] %(name)s:%(levelname)-8s - %(message)s"
_formatter = logging.Formatter(_frmt_str, datefmt="%H:%M:%S")

_stream_handler.setFormatter(_formatter)  # Add formatter to stream handler
logger.addHandler(_stream_handler)  # Add stream handler to package logger

logger.setLevel(logging.WARNING)  # Set initial logging level
logging.root.setLevel(logging.NOTSET)


# =========================================================================
# PLOTTING
# =========================================================================


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


def setup_plot(fig, ax, **kwargs):
    """Format the plot object"""
    if "xlim" in kwargs:
        ax.set_xlim(*kwargs["xlim"])
    if "ylim" in kwargs:
        ax.set_ylim(*kwargs["ylim"])
    if "xlabel" in kwargs:
        ax.set_xlabel(kwargs["xlabel"])
    if "ylabel" in kwargs:
        ax.set_ylabel(kwargs["ylabel"])
    if "grid" in kwargs:
        grid_args = kwargs["grid"]
        if isinstance(grid_args, dict):
            ax.grid(**grid_args)
        elif isinstance(grid_args, str):
            ax.grid(axis=grid_args)
        elif grid_args:
            ax.grid()
    if "title" in kwargs:
        ax.title(kwargs["title"])
    if "figsize" in kwargs:
        fig.set_size_inches(*kwargs["figsize"])
    if "legend" in kwargs:
        legend_args = kwargs["legend"]
        if isinstance(legend_args, dict):
            ax.legend(**legend_args)
        elif legend_args:
            ax.legend()
    if "tight" in kwargs:
        if kwargs["tight"]:
            fig.tight_layout()


class Plot:
    """Matplotlib.pyplot.Axes wrapper which also supports some methods of the figure.

    Parameters
    ----------
    ax : plt.Axes, optional
        Optional existing Axes instance. If ``None`` a new subplot is created.
    """

    def __init__(self, ax=None, figsize=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        self.fig, self.ax = fig, ax
        if figsize is not None:
            self.set_figsize(*figsize)

    def __getattr__(self, item):
        return getattr(self.ax, item)

    def set_equal_aspect(self):
        self.ax.set_aspect("equal")

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

    def save(self, *relpaths, dpi=600, frmt=None, rasterized=True):
        return save_figure(self.fig, *relpaths, dpi, frmt, rasterized)

    def show(self, tight=True, block=True):
        if tight:
            self.tight_layout()
        plt.show(block=block)

    @staticmethod
    def pause(interval=0.0):
        plt.pause(interval)

    def draw(self, pause=1e-10):
        self.fig.canvas.flush_events()
        plt.show(block=False)
        plt.pause(pause)

    def setup(self, **kwargs):
        """Format the plot object"""
        if "xlim" in kwargs:
            self.set_xlim(*kwargs["xlim"])
        if "ylim" in kwargs:
            self.set_ylim(*kwargs["ylim"])
        if "xlabel" in kwargs:
            self.set_xlabel(kwargs["xlabel"])
        if "ylabel" in kwargs:
            self.set_ylabel(kwargs["ylabel"])
        if "grid" in kwargs:
            grid_args = kwargs["grid"]
            if isinstance(grid_args, dict):
                self.grid(**grid_args)
            elif isinstance(grid_args, str):
                self.grid(axis=grid_args)
            elif grid_args:
                self.grid()
        if "title" in kwargs:
            self.title(kwargs["title"])
        if "figsize" in kwargs:
            self.set_figsize(*kwargs["figsize"])
        if "legend" in kwargs:
            legend_args = kwargs["legend"]
            if isinstance(legend_args, dict):
                self.legend(**legend_args)
            elif legend_args:
                self.legend()
