# -*- coding: utf-8 -*-
"""
Created on 3 Mar 2018
@author: Dylan Jones

project: cmpy
version: 1.0
"""
import os
import numpy as np
from sciutils import Path, Plot, Data, prange
from cmpy import DATA_DIR
from cmpy.tightbinding import TightBinding, TbDevice, LtData
from cmpy.tightbinding import localization_length, init_lengths, calculate_lt, localization_length_built

ROOT = Path(DATA_DIR, "Localization2", init=True)


def lognorm(array, axis=1):
    return np.mean(np.log(array), axis=axis)


# =========================================================================
# CALCULATION
# =========================================================================


def calculate_transmissions(model, folder, w_values, heights, n_avrg, soc=0, override_lengths=None):
    for h in heights:
        header = f"Height: {h}, SOC: {soc}"
        print(header + "\n" + "-"*(len(header) + 1))
        model.reshape(x=2, y=h)
        data = LtData(folder, f"disorder_h={h}.npz")
        for w in w_values:
            existing = data.get(str(w))
            if override_lengths is None:
                lengths = init_lengths(existing, model, 0, w)
            else:
                lengths = override_lengths
            arr = calculate_lt(model, lengths, w, n_avrg, existing=existing)
            data.update({str(float(w)): arr})
            data.save()
            print()
        print()


def calculate_s_basis(w_values, n_avrg=500):
    folder = ROOT.join("s-Basis", "soc=0", init=True)
    heights = [1, 4, 8, 16]
    model = TbDevice.square_basis((2, 1), basis="s")
    calculate_transmissions(model, folder, w_values, heights, n_avrg, soc=0)


def calculate_p3_basis(w_values, n_avrg=500):
    heights = [1, 4, 8, 16]
    #heights = [8]
    # soc_values = np.arange(1, 5)
    soc_values = [1]
    for soc in soc_values:
        folder = ROOT.join("p3-Basis", f"soc={soc}", init=True)
        model = TbDevice.square_basis((2, 1), basis="p3", soc=soc)
        calculate_transmissions(model, folder, w_values, heights, n_avrg, soc)


# =========================================================================
# ANALYSIS
# =========================================================================


def validate(data):
    err_keys = list()
    for k in sorted(data.keylist):
        arr = data[k]
        if not np.isfinite(arr).all():
            err_keys.append(k)
        elif np.any(arr < 0):
            err_keys.append(k)
    return err_keys


def plot_dataset(file, l0=1, show=True):
    data = LtData(file)
    plot = Plot(xlabel="$L$", ylabel=r"$\langle \ln T \rangle$")
    plot.set_title(file.basename)
    for k in sorted(data.keylist):
        lengths, trans = data.get_data(k)
        try:
            trans = lognorm(trans)
            (ll, err), y = localization_length_built(lengths, trans, l0)
            print(f"w={k} -> LL={ll}")
            plot.plot(lengths, y, color="k", ls="--")
            plot.plot(lengths, trans, label=f"w={k}")
        except ValueError as e:
            print(f"KEY {k}:", e)
    plot.legend()
    if show:
        plot.show()
    return plot


def get_loclengths(data, l0=1):
    keys = data.keylist
    n = len(keys)
    disorder = np.zeros(n)
    loclen = np.zeros(n)
    errs = np.zeros(n)
    for i, key in enumerate(keys):
        lengths, trans = data.get_data(key)
        trans = lognorm(trans)
        disorder[i] = float(key)
        loclen[i], errs[i] = localization_length(lengths, trans, l0)
    idx = np.argsort(disorder)
    return disorder[idx], loclen[idx], errs[idx]
    # return disorder, loclen, errs


def plot_localization_length(basis="s", soc=0):
    folder = Path(ROOT, f"{basis}-Basis", f"soc={soc}", init=False)
    plot = Plot()
    for file in folder.files():
        data = LtData(file)
        # plot_dataset(file, show=False)
        info = data.info()
        w, ll, err = get_loclengths(data)
        plot.plot(w, ll, label=info["h"])
    plot.legend()
    plot.show()


def plot_datapoints(file, key):
    data = LtData(file)
    lengths, arr = data.get_data(key)
    plot = Plot()
    plot.scatter(lengths, arr)
    plot.show()


def plot_heatmap():
    folder = ROOT.join("p3-Basis", "soc=1")
    for file in folder.files():
        print(file.path.filename)





def main():
    w_values = np.arange(1, 6, 0.5)
    # calculate_s_basis(w_values)
    calculate_p3_basis(w_values)
    return
    
    model = TbDevice.square_basis((59, 8), "p3", soc=1.0)
    model.set_disorder(1)
    for i in prange(10000):
        model.transmission()



if __name__ == "__main__":
    main()
