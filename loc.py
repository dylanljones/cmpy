# -*- coding: utf-8 -*-
"""
Created on 3 Mar 2018
@author: Dylan Jones

project: tightbinding
version: 1.0
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from cmpy import eta
from cmpy.tightbinding import TbDevice, sp3_basis, s_basis
from cmpy.tightbinding.loclength import Folder, LT_Data, SOC, NO_SOC, ROOT
from cmpy.tightbinding.loclength import calculate_lt, fit, get_lengths

folder = Folder(ROOT)
soc_f = Folder(SOC)
f = Folder(NO_SOC)


def print_header(txt):
    print()
    print("-" * len(txt))
    print(txt)
    print("-" * len(txt))


def init_lengths(lengths, existing, model, e, w, n_avrg=200):
    if lengths is None:
        if existing is None:
            print("Initializing lengths")
            out = get_lengths(model, e, w, n_avrg=n_avrg)
            print(f"Configured range: {out[0]}-{out[-1]}")
        else:
            out = existing[:, 0]
    else:
        out = lengths
    return out


def calculate_width_lt(heights, w, soc, lengths=None, e=0, n_avrg=250):
    omega = e + eta
    if soc is None:
        fn = f"width-e={e}-w={w}.npz"
        data = LT_Data(os.path.join(NO_SOC, fn))
        basis = s_basis()
    else:
        fn = f"width-e={e}-w={w}-soc={soc}.npz"
        data = LT_Data(os.path.join(SOC, fn))
        basis = sp3_basis(soc=soc)

    header = f"Calculating L-T-Dataset (e={e}, w={w}, soc={soc})"
    print_header(header)
    n = len(heights)
    for i in range(n):
        h = heights[i]
        key = f"h={h}"
        print(f"{i+1}/{n} (height: {h}) ")
        model = TbDevice.square((2, h), basis.eps, basis.hop)

        existing = data.get(key, None)
        sys_lengths = init_lengths(lengths, existing, model, e, w)

        arr = calculate_lt(model, sys_lengths, w, n_avrg, omega, existing=existing)
        data.update({key: arr})
        data.save()
        print()
    print()
    return data


def calculate_disorder_lt(w_values, h, soc, lengths=None, e=0, n_avrg=250):
    omega = e + eta
    # initialize basis and data
    if soc is None:
        fn = f"disord-e={e}-h={h}.npz"
        data = LT_Data(os.path.join(NO_SOC, fn))
        basis = s_basis()
    else:
        fn = f"disord-e={e}-h={h}-soc={soc}.npz"
        data = LT_Data(os.path.join(SOC, fn))
        basis = sp3_basis(soc=soc)

    # calculate dataset
    header = f"Calculating L-T-Dataset (e={e}, h={h}, soc={soc})"
    print_header(header)
    model = TbDevice.square((2, h), basis.eps, basis.hop)

    n = len(w_values)
    for i in range(n):
        w = w_values[i]
        key = f"w={w}"
        print(f"{i+1}/{n} (disorder: {w}) ")

        existing = data.get(key, None)
        sys_lengths = init_lengths(lengths, existing, model, e, w, n_avrg=500)

        arr = calculate_lt(model, sys_lengths, w, n_avrg, omega, existing=existing)
        data.update({key: arr})
        data.save()
        print()
    return data


# ====================False=========================================================


def plot_loc_length(n_fit=20):
    fig, ax = plt.subplots()
    for path in folder.files:
        data = LT_Data(path)
        print(data)
        soc, xi, err = list(), list(), list()
        w, h = float(data.info()["w"]), int(data.info()["h"])
        label = f"w={w}, height={h}"
        for k in data:
            soc.append(data.key_value(k))
            l, t = data.get_set(k, mean=True)
            popt, errs = fit(l, t, p0=[20, 1], n_fit=n_fit)
            loc_len = popt[0], errs[0]
            xi.append(loc_len[0]/h)
            err.append(loc_len[1]/h)

        ax.errorbar(soc, xi, yerr=err, label=label)
    ax.set_xlabel(r"$\lambda_{soc}$")
    ax.set_ylabel(r"$\Lambda/h$")
    ax.legend()
    plt.show()


def main():
    w = 1
    h = 1
    soc = 0

    w_values = [0.5, 0.75, 1, 1.5, 2, 2.5, 3]
    heights = [1, 2, 4]

    # calculate_width_lt(e, w, soc, heights, n_avrg=500)
    for h in heights:
        calculate_disorder_lt(w_values, h, soc, n_avrg=500)


if __name__ == "__main__":
    main()
