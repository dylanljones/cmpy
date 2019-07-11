# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2018
@author: Dylan Jones

project: ipr$
version: 1.0
"""
import os
import numpy as np
from cmpy import TbDevice, eta, DATA_DIR
from sciutils import Plot, prange, load_pkl, save_pkl, Folder



def calculate(w, file):
    model = TbDevice.square((100, 1))
    model.set_disorder(w)

    lengths = np.arange(5, 200, 5)
    n = len(lengths)
    ipr = np.zeros(n)
    for i in prange(n):
        model.reshape(lengths[i])
        ipr[i] = model.mean_ipr(n_avrg=500)

    save_pkl(os.path.join(DATA_DIR, file), [lengths, ipr], info=w)


def plot_data():
    folder = Folder(DATA_DIR)
    paths = folder.find("ipr")
    paths.sort()

    plot = Plot(xlabel="L", ylabel="IPR")
    plot.set_scales(yscale="log")
    for path in paths:
        data, w = load_pkl(path)
        plot.plot(*data, label=f"w={w}")
    plot.legend()
    plot.show()


def main():
    # calculate(0, "ipr_test1.pkl")
    # calculate(1, "ipr_test2.pkl")
    # calculate(3, "ipr_test3.pkl")
    # calculate(5, "ipr_test4.pkl")

    plot_data()




if __name__ == "__main__":
    main()
