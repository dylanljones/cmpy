# -*- coding: utf-8 -*-
"""
Created on 3 Mar 2018
@author: Dylan Jones

project: tightbinding
version: 1.0
"""
import os
import numpy as np
from cmpy import DATA_DIR, Folder, Plot
from cmpy.tightbinding import LT_Data, disorder_lt, loc_length
from cmpy.tightbinding.basis import s_basis, p3_basis, sp3_basis

ROOT = os.path.join(DATA_DIR, "Localization")


def calculate_disorder_lt(basis, w_values, h, lengths=None, e=0, n_avrg=250):
    # initialize folder
    rel_parts = [ROOT]
    if basis.n == 1:
        filename = f"disorder-e={e}-h={h}.npz"
        rel_parts.append("s-basis")
    else:
        soc = basis.soc
        filename = f"disorder-e={e}-h={h}-soc={soc}.npz"
        if basis.n == 6:
            rel_parts.append("p3-basis")
        elif basis.n == 8:
            rel_parts.append("sp3-basis")
        rel_parts.append(f"soc={soc}")
    folder = Folder(*rel_parts)
    path = folder.build_path(filename)

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


def show_loclen(relpath="p3-basis", *socs):
    folder = Folder(ROOT, relpath)
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


def calculate(n_avrg=500):
    soc_values = 0, 1, 2, 3, 4, 5
    heights = [1, 4, 8, 16]
    w_values = [2.5, 3.5, 4.5] #np.arange(16) + 1
    for soc in soc_values:
        for h in heights:
            basis = p3_basis(eps_p=0, t_pps=1, t_ppp=1, soc=soc)
            calculate_disorder_lt(basis, w_values, h, n_avrg=n_avrg)

def calculate_s_basis(n_avrg=500):
    heights = [1, 4, 8, 16]
    w_values = np.arange(16) + 1
    for h in heights:
        basis = s_basis(eps=0., t=1.)
        calculate_disorder_lt(basis, w_values, h, n_avrg=n_avrg)


def rename():
    folder = Folder(ROOT, "p3-basis")
    for path in folder.walk_files():
        root, name = os.path.split(path)
        name = name[:9] + "e=0-" + name[9:]
        os.rename(path, os.path.join(root, name))



def main():
    # calculate()
    #calculate_s_basis()
    show_loclen("p3-basis")




if __name__ == "__main__":
    main()
