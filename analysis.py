# -*- coding: utf-8 -*-
"""
Created on 3 Mar 2018
@author: Dylan Jones

project: tightbinding
version: 1.0
"""
import re
import matplotlib.pyplot as plt
from cmpy import Plot, Folder, DATA_DIR
from cmpy.tightbinding.loclength import *

ROOT = os.path.join(DATA_DIR, "localization")


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
        if mode == "lin":
            t = np.log10(t)
        n_f = int(len(l) * n_fit)

        loclen, err, fit_data = loc_length_fit(l, t, p0=[1, 1], n_fit=n_f, mode=mode)
        if norm is None:
            norm = int(k.split("=")[1])
        loclen /= norm
        err /= norm
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


def loclen_plotter(ax, data, log=False):
    height = int(data.info()["h"])
    disorder = list()
    loclen = list()
    errs = list()
    for k in data:
        l, t = data.get_set(k, mean=True)
        n_f = int(len(l) * 1)
        w = data.key_value(k)
        disorder.append(w)
        ll, err = loc_length(l, np.log10(t), [1, 1], n_f)
        loclen.append(ll)
        errs.append(err)

    idx = np.argsort(disorder)
    disorder = np.array(disorder)[idx]
    loclen = np.array(loclen)[idx] / height
    errs = np.array(errs)[idx] / height

    if log:
        loclen = np.log10(loclen)
        errs = np.log10(errs)

    # ax.errorbar(disorder, loclen, yerr=errs, label=f"$M={height}$")
    ax.semilogy(disorder, loclen, label=f"$M={height}$")


def sort_paths(paths, query="h="):
    heights = [int(re.search(query + "(\d+)", p).group(1)) for p in paths]
    idx = np.argsort(heights)
    return [paths[i] for i in idx]



def plot_wl(path, soc=None, show=True):
    f = Folder(path)
    plot = Plot()
    if soc is None:
        paths = [x for x in f.find("disord-") if "soc=" not in x]
    else:
        paths = f.find("disord-", f"soc={soc}")
        plot.set_title(r"$\lambda_{SOC}=$" + f"${soc}$")
    plot.set_labels("$w$", r"$\xi / M$")
    paths = sort_paths(paths)

    for i, path in enumerate(paths):
        data = LT_Data(path)
        data.sort_all()
        data.save()
        try:
            loclen_plotter(plot.ax, data)
        except:
            pass
    plot.legend()
    if show:
        plot.show()


def delete_keys(path, key):
    folder = Folder(path)
    for path in folder.files:
        data = LT_Data(path)
        if data.get(key, None) is not None:
            del data[key]
        data.save()


def plot_data(path):
    f = Folder(path)
    for path in f.files:
        data = LT_Data(path)
        print(data.info_str())
        plot = Plot()
        for k in data:
            l, t = data.get_set(k, mean=True)
            try:
                t = np.log10(t)
                #print(loc_length(l, t))
            except Exception as e:
                pass
            plot.plot(l, t)
        plt.show()


def plot_trans(path):
    data = LT_Data(path)
    print(data.info_str())
    plot = Plot()
    for k in data:
        l, t = data.get_set(k, mean=True)
        t = np.log10(t)
        plot.plot(l, t, label=k)
    plt.legend()
    plt.show()


def main():
    # plot_trans(os.path.join(P3_PATH, f"disord-e=0-h=4-soc=2.npz"))
    # delete_keys(P3_PATH, "w=1.5")
    # plot_data(P3_PATH)
    # plot_all_lt("disord-")

    plot_wl(S_PATH, None, False)
    # plot_wl(P3_PATH, 0, False)
    plot_wl(P3_PATH, 1, False)
    plot_wl(P3_PATH, 2, False)
    # plot_wl(SP3_PATH, 0, False)
    # plot_wl(SP3_PATH, 1, False)


    plt.show()


if __name__ == "__main__":
    main()
