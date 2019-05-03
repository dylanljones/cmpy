# -*- coding: utf-8 -*-
"""
Created on 26 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
import re
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cmpy import *
from cmpy.tightbinding import TbDevice, LT_Data
from cmpy.tightbinding import disorder_lt, calculate_lt, loc_length
from cmpy.tightbinding.basis import *

ROOT = os.path.join(DATA_DIR, "Tests", "Localization")


def sort_paths(paths, query="h="):
    heights = [int(re.search(query + r"(\d+)", p).group(1)) for p in paths]
    idx = np.argsort(heights)
    return [paths[i] for i in idx]


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


def search_string_value(string, header):
    return re.search(header + r"(\d+)", string).group(1)


# =============================================================================


def read_loclen_data(subfolder):
    data_list = list()
    for path in subfolder.listfiles():
        data = LT_Data(path)
        sort_keys(data)
        h = data.info()["h"]
        w, ll, errs = list(), list(), list()
        for k in data:
            l, t = data.get_set(k, mean=True)
            w.append(data.key_value(k))
            lam, lam_err = loc_length(l, np.log10(t))
            ll.append(lam)
            errs.append(lam_err)
        w = np.array(w)

        # Normalizing data
        ll = np.array(ll) / h
        errs = np.array(errs) / h

        data_list.append((h, w, ll, errs))
    return data_list


def show_loclen(*socs):
    folder = Folder(ROOT, "p3-basis_2")
    for subfolder in folder.subfolders():
        if len(socs) and not any([f"soc={s}" in subfolder.name for s in socs]):
            continue
        data_list = read_loclen_data(subfolder)

        plot = Plot()
        plot.set_scales(yscale="log")
        plot.set_title(subfolder.name)
        plot.set_labels(r"Disorder $w$", r"$\xi / M$")   #r"$\log_{10}(\xi / M)$")
        for h, w, ll, errs in sorted(data_list, key=lambda x: x[0]):
            plot.ax.errorbar(w, ll, yerr=errs, label=f"M={h:.0f}")
        plot.legend()
        plot.tight()
    plot.show()


def plot_all_loclen(*socs):
    folder = Folder(ROOT, "p3-basis")
    plot = Plot()
    plot.set_scales(yscale="log")
    plot.set_labels(r"Disorder $w$", r"$\xi / M$")
    styles = ["-", "--", "-.", ":"]
    i = 0
    for subfolder in folder.subfolders():
        soc = float(subfolder.name.replace("soc=", ""))
        if len(socs) and soc not in socs:
            continue
        data_list = read_loclen_data(subfolder)
        col = f"C{i}"
        j = 0

        for h, w, ll, errs in sorted(data_list, key=lambda x: x[0]):
            label = f"soc={soc}" if j == 0 else None
            plot.ax.errorbar(w, ll, yerr=errs, color=col, ls=styles[j], label=label)
            j += 1
        i += 1
    plot.legend()
    plot.tight()
    plot.show()


def calculate_test_data(n_avrg=250):
    fpath = "p3-basis_2"
    soc_values = 1, 2, 3, 4, 5, 7, 10
    heights = [1, 4, 8, 16]
    w_values = np.arange(16) + 1
    for soc in soc_values:
        for h in heights:
            try:
                # Init path and model-config
                folder = Folder(ROOT, fpath, f"soc={soc}")
                path = os.path.join(folder.path, f"disorder-h={h}-soc={soc}.npz")
                basis = p3_basis(eps_p=0, t_pps=1, t_ppp=1, soc=soc)
                # calculate
                disorder_lt(path, basis, h, w_values, n_avrg=n_avrg)
            except Exception as e:
                print("ERROR")
                print(e)
                print()


def main():
    calculate_test_data(500)
    #plot_all_loclen()
    #show_loclen(0, 1, 2, 3, 4, 5)


if __name__ == "__main__":
    main()
