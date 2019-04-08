# -*- coding: utf-8 -*-
"""
Created on 3 Mar 2018
@author: Dylan Jones

project: tightbinding
version: 1.0
"""
import os
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from cmpy import Plot
from cmpy.tightbinding.loclength import *

folder = Folder(ROOT)


def _label(key, loclength=None, err=None, n_avrg=None):
    key_txt, key_val = key.split("=")
    label = f"{key_txt}={float(key_val):.1f}"
    if loclength and err:
        label += rf", $\Lambda ={loclength:.1f} \pm {err:.1f}$"
    if n_avrg:
        label += f" (n={n_avrg})"
    return label


def plot_lt(data, n_fit=1., mode="lin", show=True):
    plot = Plot()
    plot.set_title(data.info_str())
    ax = plot.ax
    if mode == "exp":
        ax.set_yscale("log")
        ylabel = r"$T/T_0$"
    else:
        ylabel = r"$\log(T/T_0)$"

    if "h" in list(data.keys())[0]:
        norm = None
    else:
        norm = data.info()["h"]

    i, xmax = 0, 0
    for k in data:
        col = f"C{i}"
        l, t = data.get_set(k, mean=True)
        if norm is None:
            h = int(k.split("=")[1])
            t /= h


        if mode == "lin":
            t = np.log10(t)
        n_f = int(len(l) * n_fit)
        loclen, err, fit_data = loc_length_fit(l, t, p0=[20, 1], n_fit=n_f, mode=mode)
        # loclen /= norm
        # err /= norm
        n_avrg = data.n_avrg(k)
        label = _label(k, loclen, err, n_avrg=n_avrg)
        ax.plot(l, t, label=label, color=col)
        ax.plot(*fit_data, color="k", ls="--")

        i += 1
        xmax = max(xmax, max(l))

    plot.set_limits((0, xmax + 10))
    plot.set_labels("N", ylabel)
    plot.legend()
    if show:
        plot.show()



def main():
    path = folder.find("width-", "w=2")[0]
    data = LT_Data(path)
    plot_lt(data)
    #plot_lt(data, n_fit=1, show=False, norm=1)

    path = folder.find("disord-", "h=1")[0]
    data = LT_Data(path)
    plot_lt(data, n_fit=0.5)
    # for i in range(3):
    #     plot_curv(data, i)

    plt.show()


if __name__ == "__main__":
    main()
