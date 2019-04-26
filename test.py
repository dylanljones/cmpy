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
from cmpy.tightbinding import *
from cmpy.tightbinding.basis import *

TEST_DIR = os.path.join(DATA_DIR, "Tests")
TEST_LOC = os.path.join(TEST_DIR, "Localization")
create_dir(TEST_LOC)


def calculate_test_data(soc_values, heights, w_values, n_avrg=100):
    for soc in soc_values:
        for h in heights:
            # Init path and model-config
            dirpath = os.path.join(TEST_LOC, "p3-basis", f"soc={soc}")
            create_dir(dirpath)
            path = os.path.join(dirpath, f"test-h={h}-soc={soc}.npz")
            basis = p3_basis(eps_p=0, t_pps=1, t_ppp=1, soc=soc)
            # calculate
            disorder_lt(path, basis, h, w_values, n_avrg=n_avrg)


def sort_files():
    new_root = os.path.join(TEST_DIR, "Localization", "p3-basis")
    for fn in os.listdir(PROJECT_DIR):
        if fn.endswith(".npz"):
            path = os.path.join(PROJECT_DIR, fn)
            soc = int(search_string_value(fn, "soc="))
            #print(soc, h)
            new_dir = os.path.join(new_root, f"soc={soc}")
            create_dir(new_dir)

            new_path = os.path.join(new_dir, fn)
            shutil.copyfile(path, new_path)


def sort_paths(paths, query="h="):
    heights = [int(re.search(query + "(\d+)", p).group(1)) for p in paths]
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
    return re.search(header + "(\d+)", string).group(1)


def show_loclen(*socs):
    root = os.path.join(TEST_DIR, "Localization", "p3-basis")
    for dirname in os.listdir(root):
        dirpath = os.path.join(root, dirname)
        if len(socs) and not any([f"soc={s}" in dirname for s in socs]):
            continue

        data_list = list()
        for path in list_files(dirpath):
            data = LT_Data(path)
            sort_keys(data)
            h = data.info()["h"]
            w, ll = list(), list()
            for k in data:
                l, t = data.get_set(k, mean=True)
                w.append(data.key_value(k))
                ll.append(loc_length(l, np.log10(t))[0])
            data_list.append((h, w, ll))

        plot = Plot()
        plot.set_title(dirname)
        for h, w, ll in sorted(data_list, key=lambda x: x[0]):
            plot.plot(w, ll, label=f"M={h}")
        plot.legend()
    plot.show()



def conductance(t):
    return t / (1 - t)


def resistance(g):
    return 1 + 1/g


def beta_func(g):
    return -(1 + g) * np.log(resistance(g))


def beta_strong(g, g0):
    return np.log(g / g0)


def plot_scaling():
    model = TbDevice.square((2, 1))
    model.set_disorder(0.5)
    lengths = np.arange(2, 200, 5)

    t0 = model.normal_transmission()
    g0 = conductance(t0)

    t = model.transmission_loss(lengths, n_avrg=200, flatten=True)
    g = conductance(t)


    plot = Plot()
    plot.lines(x=0, y=0, color="0.5", lw=0.5)
    plot.set_limits(xlim=(-4, 4), ylim=(-4, 1))
    x = np.log(g)
    idx = np.argsort(x)
    plot.ax.plot(x[idx], beta_func(g)[idx])
    plot.ax.plot(g, beta_strong(g, g0))
    plot.show()



def main():
    #plot_scaling()
    #return
    soc_values = 1, 2, 3, 4, 5, 10
    heights = [1, 4, 8, 16]
    w_values = [1, 2, 3, 4, 6, 8, 12, 16]

    #calculate_test_data(soc_values, heights, w_values)

    show_loclen()




if __name__ == "__main__":
    main()
