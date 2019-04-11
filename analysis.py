# -*- coding: utf-8 -*-
"""
Created on 3 Mar 2018
@author: Dylan Jones

project: tightbinding
version: 1.0
"""
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
            print(h)
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


def plot_all_lt(*args):
    for path in folder.find(*args):
        data = LT_Data(path)
        plot_lt(data, show=False)
    plt.show()


def sort_keys(data):
    keys, values = list(), list()
    for k, v in data.items():
        keys.append(k)
        values.append(v)
    key_vals = [data.key_value(k) for k in keys]
    idx = np.argsort(key_vals)

    data.clear()
    for i in idx:
        data.update({keys[i]: values[i]})
    data.save()


def loclen_plotter(ax, i, data, log=False):
    height = data.info()["h"]

    disorder = list()
    loclen = list()
    errs = list()
    for k in data:
        l, t = data.get_set(k, mean=True)
        n_f = int(len(l) * 1)
        w = data.key_value(k)
        disorder.append(w)
        ll, err = loc_length(l, np.log10(t), [20, 1], n_f)
        loclen.append(ll)
        errs.append(err)

    loclen = np.array(loclen) / height
    errs = np.array(errs) / height
    if log:
        loclen = np.log10(loclen)
        errs = np.log10(errs)

    ax.errorbar(disorder, loclen, yerr=errs, label=f"$M={height}$")
    # ax.semilogy(disorder, loclen, label=f"$M={height}$")


def plot_wl(soc=0, show=True):
    plot = Plot()
    plot.set_title(r"$\lambda_{SOC}=$" + f"${soc}$")
    plot.set_labels("$w$", r"$\xi / M$")
    for i, path in enumerate(folder.find("disord-", f"soc={soc}")):
        data = LT_Data(path)
        loclen_plotter(plot.ax, i, data)
    plot.legend()
    if show:
        plot.show()


def main():
    # plot_all_lt("disord-", "soc=")
    plot_wl(0, False)
    plot_wl(1)


if __name__ == "__main__":
    main()
