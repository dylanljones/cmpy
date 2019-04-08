# -*- coding: utf-8 -*-
"""
Created on 3 Mar 2018
@author: Dylan Jones

project: tightbinding
version: 1.0
"""
import os
from time import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from cmpy import eta, time_str
from cmpy.tightbinding import TbDevice, sp3_basis, s_basis
from cmpy.tightbinding.loclength import Folder, LT_Data, SOC, NO_SOC, ROOT
from cmpy.tightbinding.loclength import update_lt, calculate_lt, fit, get_lengths

folder = Folder(ROOT)
soc_f = Folder(SOC)
f = Folder(NO_SOC)


def init_lengths(lengths, existing, model, e, w):
    if lengths is None:
        if existing is None:
            l = get_lengths(model, e, w)
        else:
            l = existing[:, 0]
    else:
        l = lengths
    return l



def calculate_soc_lt(e, h, w, lengths, soc_vals, n_avrg=250):
    omega = e + eta
    data = LT_Data.new(SOC, e, w, h)
    print(f"Calculating L-T-Dataset (e={e}, w={w}, h={h})")
    print("-"*40)
    n = len(soc_vals)
    for i in range(n):
        soc = soc_vals[i]
        key = f"soc={soc}"
        pre_txt = f"{i+1}/{n} (Soc: {soc}) "

        basis = sp3_basis(soc=soc)
        model = TbDevice.square((2, h), basis.eps, basis.hop)

        existing = data.get(key, None)
        lengths = init_lengths(lengths, existing, model, e, w)

        arr = calculate_lt(model, omega, lengths, w, n_avrg, existing=existing, pre_txt=pre_txt)
        data.update({key: arr})
        data.save()
    print()
    return data


def calculate_disorder_lt(e, h, w_values, lengths, n_avrg=250):
    omega = e + eta
    print(f"Calculating L-T-Dataset (e={e}, h={h})")
    print("-"*40)
    model = TbDevice.square((2, h))
    data = LT_Data(os.path.join(NO_SOC, f"disord-e={e}-h={h}.npz"))
    n = len(w_values)
    for i in range(n):
        w = w_values[i]
        key = f"w={w}"
        pre_txt = f"{i + 1}/{n} (w: {w}) "

        existing = data.get(key, None)
        lengths = init_lengths(lengths, existing, model, e, w)

        arr = calculate_lt(model, omega, lengths, w, n_avrg, existing=existing, pre_txt=pre_txt)
        data.update({key: arr})
        data.save()
    return data



def calculate_width_lt(e, w, soc, heights, lengths=None, n_avrg=250):
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
    print(header)
    print("-" * len(header))
    n = len(heights)
    for i in range(n):
        h = heights[i]
        key = f"h={h}"
        print(f"{i+1}/{n} (height: {h}) ")
        model = TbDevice.square((2, h), basis.eps, basis.hop)

        existing = data.get(key, None)
        l = init_lengths(lengths, existing, model, e, w)

        arr = calculate_lt(model, omega, l, w, n_avrg, existing=existing)
        data.update({key: arr})
        data.save()
        print()
    print()
    return data


def update(data, n_avrg=500, new_lengths=None):
    info = data.info()
    e = float(info["e"])
    w = float(info["w"])
    h = int(info["h"])
    n_keys = len(data)
    print(f"Updating L-T-Dataset (e={e}, w={w}, h={h})")
    print("-"*40)
    for i, key in enumerate(data):
        soc = data.key_value(key)
        pre_txt = f"{i+1}/{n_keys} {data.filename} "
        basis = sp3_basis(soc=soc)
        model = TbDevice.square((1, h), basis.eps, basis.hop)
        arr = update_lt(model, e + eta, w, data[key], n_avrg, new_lengths, pre_txt)
        data[key] = arr
        data.save()
    print()


def update_all(n_avrg=500):
    for file in folder.files:
        update(LT_Data(file), n_avrg=n_avrg)


# =============================================================================


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
    e = 0
    w = 2
    soc = 0
    h = 4
    heights = [1, 4, 8, 16]
    soc_vals = [0, 0.5, 1, 2]
    lengths = np.arange(0, 200, 5) + 5

    calculate_width_lt(e, w, None, heights, n_avrg=500)

    # calculate_disorder_lt(e, h=1, w_values=[0.5, 1, 2], lengths=np.arange(0, 400, 5) + 5, n_avrg=500)
    # calculate_soc_lt(e, h, 0.5, lengths, soc_vals, n_avrg=250)
    # calculate_soc_lt(e, h, 1, lengths, soc_vals, n_avrg=250)
    # plot_datasets(folder, ["soc-", "h=4"], n_fit=10)
    # plot_loc_length(20)


if __name__ == "__main__":
    main()
