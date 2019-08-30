# -*- coding: utf-8 -*-
"""
Created on 3 Mar 2018
@author: Dylan Jones

project: tightbinding
version: 1.0
"""
import os
import numpy as np
from sciutils import Path, Plot
from cmpy import DATA_DIR
from cmpy.tightbinding import LT_Data, localization_length


ROOT = os.path.join(DATA_DIR, "Localization")


def calculate_disorder_lt(basis, w_values, h, lengths=None, e=0, n_avrg=250):
    # initialize folder
    rel_parts = [ROOT]
    if basis.n == 1:
        filename = f"disorder-e={e}-h={h}.npz"
        rel_parts.append("s-basis")
    else:
        soc = basis.soc
        if soc is None:
            soc = 0
        filename = f"disorder-e={e}-h={h}-soc={soc}.npz"
        if basis.n == 6:
            rel_parts.append("p3-basis")
        elif basis.n == 8:
            rel_parts.append("sp3-basis")
        rel_parts.append(f"soc={soc}")
    folder = Path(*rel_parts)
    path = folder.add(filename)

    # calculate the transmission data
    try:
        disorder_lt(path, basis, h, w_values, lengths=lengths, e=e, n_avrg=n_avrg)
    except Exception as e:
        print("ERROR")
        print(e)
        print()


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


def read_loclen_data(subfolder):
    data_list = list()
    for path in subfolder.files():
        data = LT_Data(path)
        sort_keys(data)
        h = data.info()["h"]
        w, ll, errs = list(), list(), list()
        for k in data:
            l, t = data.get_set(k, mean=False)
            w.append(data.key_value(k))
            try:
                t = np.mean(np.log(t), axis=1)
                lam, lam_err = localization_length(l, t)
                ll.append(lam / h)
                errs.append(lam_err)
            except Exception as e:
                print(k)

        w = np.array(w)

        # Normalizing data
        ll = np.array(ll)
        errs = np.array(errs)

        data_list.append((h, w, ll, errs))
    return data_list


def show_loclen(relpath="p3-basis", *socs):
    folder = Path(ROOT, relpath)
    plot = Plot()
    for subfolder in folder.dirs(deep=True):
        if len(socs) and not any([f"soc={s}" in subfolder.name for s in socs]):
            continue
        data_list = read_loclen_data(subfolder)

        plot.set_scales(yscale="log")
        plot.set_title(subfolder.name)
        plot.set_labels(r"Disorder $w$", r"$\xi / M$")   #r"$\log_{10}(\xi / M)$")
        for h, w, ll, errs in sorted(data_list, key=lambda x: x[0]):
            plot.plot(w, ll, label=f"M={h:.0f}")
            # plot.ax.errorbar(w, ll, yerr=errs, label=f"M={h:.0f}")
        plot.legend()
        plot.tight()
    plot.show()


def show_dataset(data):
    plot = Plot(xlabel="$L$", ylabel=r"$\langle \ln T \rangle$")
    for k in data:
        lengths, trans = data.get_set(k, mean=False)
        trans = np.mean(np.log(trans), axis=1)
        plot.plot(lengths, trans)
    return plot
    #plot.show()


def calculate_s_basis(n_avrg=500):
    heights = [1, 4, 8, 16]
    w_values = np.arange(10) + 1
    for h in heights:
        basis = s_basis(eps=0., t=1.)
        calculate_disorder_lt(basis, w_values, h, n_avrg=n_avrg)


def calculate(n_avrg=500):
    soc_values = 0, 1, 2, 3, 4
    heights = [1, 4, 8, 16]
    w_values = np.arange(10) + 1
    for soc in soc_values:
        for h in heights:
            basis = p3_basis(eps_p=0, t_pps=1, t_ppp=1, soc=soc)
            calculate_disorder_lt(basis, w_values, h, n_avrg=n_avrg)


def calculate_single_soc(n_avrg=500):
    soc = 1
    heights = [1, 4, 8, 16]
    w_values = np.arange(3, 6, 0.25)
    for h in heights:
        basis = p3_basis(eps_p=0, t_pps=1, t_ppp=1, soc=soc)
        calculate_disorder_lt(basis, w_values, h, n_avrg=n_avrg)


def main():
    # save_localization_data(ROOT)

    # calculate()
    # calculate_single_soc()
    # calculate_s_basis()
    show_loclen("sp3-basis", 1, 2)


if __name__ == "__main__":
    main()
