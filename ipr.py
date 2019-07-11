# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2018
@author: Dylan Jones

project: ipr$
version: 1.0
"""
import os
import numpy as np
from cmpy2 import Plot, prange, eta, Progress, DATA_DIR
from cmpy2.tightbinding import TbDevice
from cmpy2.tightbinding import LoclenData, save_localization_data, get_loclen_files

ROOT = os.path.join(DATA_DIR, "Localization")


def mean_ipr(model, omega, n_avrg=200):
    tmp = np.zeros(n_avrg)
    for i in range(n_avrg):
        tmp[i] = model.inverse_participation_ratio(omega)
    return np.mean(tmp)


def plot_ipr(model):
    n = 20
    w_values = [0, 1, 2]

    e_vals = np.linspace(0, 2, n)
    ipr_vals = np.zeros(n)

    plot = Plot()
    for i in range(len(w_values)):
        w = w_values[i]
        model.set_disorder(w)
        for j in prange(n):
            omega = e_vals[j] + eta
            ipr_vals[j] = mean_ipr(model, omega, 300)
        plot.plot(e_vals, ipr_vals, label=f"w={w}")
    plot.legend()
    plot.show()


def update_mean(i, mean, x):
    n = i+1
    return x if n == 1 else mean + (x - mean) / n


def ipr_loclen(model):
    n_avrg = 100
    w = 1
    ipr = 0
    model.set_disorder(w)
    with Progress(total=n_avrg) as p:
        for i in range(n_avrg):
            p.set_description(f"IPR={ipr:.5f}")
            p.update()
            x = model.inverse_participation_ratio(eta)
            ipr = update_mean(i, ipr, x)
    print(ipr)


def ipr_length_curve(model, lengths, n_avrg=100):
    n = len(lengths)
    ipr = np.zeros(n)
    tmp = np.zeros(n_avrg)
    with Progress(total=n * n_avrg) as p:
        for i in range(n):
            model.reshape(lengths[i])
            for j in range(n_avrg):
                p.update()
                tmp[j] = model.inverse_participation_ratio()
            ipr[i] = np.mean(tmp)
    return ipr


def check_ipr(model):
    lengths = np.arange(100, 200+1, 5)
    n_avrg = 100

    plot = Plot()

    ipr = ipr_length_curve(model, lengths)
    plot.plot(lengths, ipr, label="w=0")
    ymax = max(ipr)*1.2

    model.set_disorder(1)
    ipr = ipr_length_curve(model, lengths)
    plot.plot(lengths, ipr, label="w=1")
    ymax = max(max(ipr)*1.2, ymax)

    plot.set_limits(ylim=(0, ymax))
    plot.legend()

    plot.show()


def get_loclen(basis=None, soc=None, e=None):
    files = list()
    for path in get_loclen_files(ROOT, basis, soc, e):
        files.append(LoclenData(path))
    return files


def compare_loclen_to_ipr():
    model = TbDevice.square((100, 1))
    print("Reading data")
    data = get_loclen("s-basis")[0]
    w_vals, ll, llerr = data["h=1"]
    n = len(w_vals)
    faktor = list()
    for i in prange(n):
        w = w_vals[i]
        model.set_disorder(w)
        ipr = mean_ipr(model, eta)
        f = ll[i] / ipr

        faktor.append(f)

    plot = Plot()
    plot.plot(np.log10(faktor))
    plot.show()




def main():
    model = TbDevice.square((100, 1))
    # model.set_disorder(1)
    # ipr_loclen(model)
    compare_loclen_to_ipr()
    # check_ipr(model)
    #plot_ipr(model)
    # ipr_map(model)


if __name__ == "__main__":
    main()
