# -*- coding: utf-8 -*-
"""
Created on 3 Mar 2018
@author: Dylan Jones

project: cmpy
version: 1.0
"""
import numpy as np
from sciutils import Path, Plot, Data, prange
from cmpy import DATA_DIR
# from cmpy.models import TightBinding
from cmpy.transport import TbDevice, LtData
from cmpy.transport import localization_length, init_lengths, calculate_lt, localization_length_built

ROOT = Path(DATA_DIR, "Localization2", init=True)


def lognorm(array, axis=1):
    return np.mean(np.log(array), axis=axis)


# =========================================================================
# CALCULATION
# =========================================================================


def calculate_transmissions(model, folder, w_values, heights, n_avrg, soc=0, override_lengths=None,
                            w_thresh=3.0):
    for h in heights:

        header = f"Height: {h}, SOC: {soc}"
        print(header + "\n" + "-"*(len(header) + 1))

        model.reshape(x=2, y=h)
        data = LtData(folder, f"disorder_h={h}.npz")
        for w in w_values:
            existing = data.get(str(w))
            if override_lengths is not None:
                lengths = override_lengths
            elif w >= w_thresh:
                lengths = np.arange(50, 150, 50)
            else:
                try:
                    lengths = init_lengths(existing, model, 0, w)
                except ValueError as e:
                    print(e)
                    print("Using default lengths: 50-150")
                    lengths = np.arange(50, 150, 50)
            txt = f"W={w} "
            arr = calculate_lt(model, lengths, w, n_avrg, pre_txt=txt, existing=existing)
            data.update({str(float(w)): arr})
            data.save()
            print()
        print()


def calculate_s_basis(w_values, n_avrg=500):
    folder = ROOT.join("s-Basis", "soc=0", init=True)
    heights = [1, 4, 8, 16]
    model = TbDevice.square_basis((2, 1), basis="s")
    calculate_transmissions(model, folder, w_values, heights, n_avrg, soc=0)


def calculate_p3_basis(w_values, heights, soc_values, n_avrg=500, override_lengths=None):
    header = f"Calculating Data: SOC: {soc_values}, Disorder: {min(w_values)}-{max(w_values)}"
    print("\n" + header + "\n" + "="*len(header) + "\n")

    for soc in soc_values:
        soc_str = str(soc).replace(".", "_")
        folder = ROOT.join("p3-Basis", f"soc={soc_str}", init=True)
        model = TbDevice.square_basis((2, 1), basis="p3", soc=soc)
        calculate_transmissions(model, folder, w_values, heights, n_avrg, soc, override_lengths)


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


def plot_dataset(data, l0=1, show=True):
    plot = Plot(xlabel="$L$", ylabel=r"$\langle \ln T \rangle$")
    plot.set_title(data.path.relpath(ROOT))
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


def localization_curves(basis="s", soc=0.):
    folder = Path(ROOT, f"{basis}-Basis", f"soc={soc}".replace(".", "_"), init=False)
    lines = list()
    for file in folder.files():
        data = LtData(file)
        w, ll, err = data.localization_curve(l0=1.)
        h = data.info()["h"]
        lines.append([h, w, ll/h, err/h])
    return sorted(lines, key=lambda x: x[0])


def plot_localization_length(basis="s", soc=0., show=True):
    lines = localization_curves(basis, soc)
    plot = Plot()
    plot.set_scales(yscale="log")
    for h, w, ll, err in lines:
        plot.errorplot(w, ll, err, label=h)
    plot.legend()
    plot.set_title(f"{basis}-Basis, SOC: {soc}")
    if show:
        plot.show()
    return plot


def plot_heatmap(h=4):
    files = sorted(ROOT.join("p3-Basis").search(f"h={h}.npz"))
    plot = Plot()
    for i, file in enumerate(files):
        data = LtData(file)
        soc = data.get_soc()
        w, ll, err = data.localization_curve()
        ll = ll / h
        plot.plot(w, ll, label=f"SOC={soc}")
    plot.legend()
    plot.show()


def main():
    heights = [1, 4, 8, 16]
    w_values = np.arange(1.0, 6, 0.5)
    soc_values = [0.5, 1.0, 1.5]
    # soc_values = [2.0, 2.5, 3.0]

    calculate_p3_basis(w_values, heights, soc_values)
    # data = LtData.find(ROOT, "p3", 0, 1)
    # plot_dataset(data, show=False)
    plot_localization_length("p3", soc=2.0)


if __name__ == "__main__":
    main()
