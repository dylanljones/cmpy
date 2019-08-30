# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2018
@author: Dylan Jones

project: cmpy
version: 1.0
"""
import os
import numpy as np
import cmpy.tightbinding as tb
from cmpy import DATA_DIR
from cmpy.tightbinding import TightBinding, eta
from cmpy.tightbinding import configure_p3_basis, configure_sp3_basis
from sciutils import Plot, Progress, load_pkl, save_pkl, Path

IPR_DIR = Path(DATA_DIR, "ipr")


def plot_data(*subdirs):
    folder = IPR_DIR
    if subdirs:
        folder = folder.join(*subdirs)
    plot = Plot(xlabel="L", ylabel=r"$\langle IPR\rangle$")
    plot.set_scales(yscale="log")
    for path in sorted(folder.search("ipr_")):
        wstr, hstr = path.basename.split("_")[1:]
        lengths, array = load_pkl(path)[0]
        ipr = np.mean(array, axis=1)
        plot.plot(lengths, ipr, label=f"{wstr}, {hstr}")
    plot.legend()
    plot.show()


def calculate_ipr(model, lengths, w, file, n_avrg=100, n_save=5):
    model.set_disorder(w)
    if file.exists:
        _lengths, array = load_pkl(file)[0]
        if np.all(_lengths == lengths):
            return
    else:
        _lengths = []
    n = len(lengths)
    ipr = np.zeros((n, n_avrg))
    with Progress(total=n*n_avrg, header=f"Calculating IPR (h={model.lattice.shape[1]}, w={w})") as p:
        for i in range(n):
            length = lengths[i]
            idx = np.where(_lengths == length)[0]
            if idx:
                ipr[i] = array[idx]
            else:
                p.set_description(f"L={lengths[i]}")
                model.reshape(lengths[i])
                for j in range(n_avrg):
                    p.update()
                    model.shuffle()
                    ipr[i, j] = model.inverse_participation_ratio(omega=eta)
            if i % n_save == 0:
                data = [lengths[:i], ipr[:i]]
                save_pkl(file, data)

    save_pkl(file, [lengths, ipr])
    return ipr


def calculate(w, soc, heights=[1, 5, 10], n_avrg=500):
    w = 1
    soc = 1
    basis = "p3"
    lengths = np.arange(5, 150, 5)

    root = Path(IPR_DIR, f"{basis}-Basis", f"soc={soc}", init=True)
    for h in [1, 5, 10, 15]:
        file = root.join(f"ipr_h={h}_w={w}.pkl")
        model = TightBinding.square_basis((5, h), basis=basis, soc=soc)
        calculate_ipr(model, lengths, w, file, n_avrg=n_avrg)



def main():
    # calculate(w=1, soc=1)
    # calculate(w=1, soc=0)
    plot_data("p3-Basis")


if __name__ == "__main__":
    main()
