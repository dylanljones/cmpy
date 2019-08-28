# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2018
@author: Dylan Jones

project: cmpy
version: 1.0
"""
import os
import numpy as np
from cmpy import DATA_DIR
from cmpy.tightbinding import TightBinding, eta
from cmpy.tightbinding import configure_p3_basis, configure_sp3_basis
from sciutils import Plot, Progress, prange, load_pkl, save_pkl, Path

folder = Path(DATA_DIR, "ipr")


def calculate_ipr(model, lengths, w, n_avrg=500):
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
    return ipr


def plot_data():
    paths = folder.search("ipr")
    plot = Plot(xlabel="L", ylabel=r"$\langle IPR\rangle$")
    plot.set_scales(yscale="log")
    for path in sorted(paths):
        (lengths, array), w = load_pkl(path)
        ipr = np.mean(array, axis=1)
        plot.plot(lengths, ipr, label=f"w={w}")
    plot.legend()
    plot.show()


def calculate():
    h = 4
    soc = 1
    w = 1
    lengths = np.arange(5, 150, 5)

    model = TightBinding(np.eye(2))
    configure_p3_basis(model, soc=soc)

    f = folder.makedirs(f"soc={soc}")
    file = f"ipr_h={h}_w={w}.pkl"
    model.build((5, h))
    model.set_disorder(w)

    ipr = calculate_ipr(model, lengths, w, n_avrg=500)
    save_pkl(os.path.join(f, file), [lengths, ipr], info=w)


def main():
    calculate()

    plot_data()


if __name__ == "__main__":
    main()
