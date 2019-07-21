# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2018
@author: Dylan Jones

project: cmpy
version: 1.0
"""
import os
import numpy as np
from cmpy import TbDevice, TightBinding, eta, DATA_DIR
from sciutils import Plot, Progress, prange, load_pkl, save_pkl, Folder

folder = Folder(DATA_DIR, "ipr2")


def calculate(lengths, w, height=1, n_avrg=500):
    file = f"ipr_test{w}.pkl"
    f = folder.subfolder(f"h={height}")
    model = TightBinding.square((100, height))
    model.set_disorder(w)

    n = len(lengths)
    ipr = np.zeros((n, n_avrg))
    with Progress(total=n*n_avrg, header="Calculating IPR") as p:
        for i in range(n):
            p.set_description(f"L={lengths[i]}")
            model.reshape(lengths[i])
            for j in range(n_avrg):
                p.update()
                model.shuffle()
                ipr[i, j] = model.inverse_participation_ratio(omega=eta)
    save_pkl(os.path.join(f, file), [lengths, ipr], info=w)


def plot_data():
    paths = folder.find("ipr")
    plot = Plot(xlabel="L", ylabel=r"$\langle IPR\rangle$")
    plot.set_scales(yscale="log")
    for path in sorted(paths):
        (lengths, array), w = load_pkl(path)
        ipr = np.mean(array, axis=1)
        plot.plot(lengths, ipr, label=f"w={w}")
    plot.legend()
    plot.show()


def main():
    h = 1
    lengths = np.arange(5, 200, 5)
    # calculate(lengths, 0, h, n_avrg=1)
    # calculate(lengths, 1, h)
    # calculate(lengths, 2, h)
    # calculate(lengths, 3)
    # calculate(lengths, 4)
    plot_data()


if __name__ == "__main__":
    main()
