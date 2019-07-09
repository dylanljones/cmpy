# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2018
@author: Dylan Jones

project: ipr$
version: 1.0
"""
import numpy as np
from cmpy import Plot, prange, Progress, eta
from cmpy.tightbinding import TbDevice


def ipr_curve(model, w_values, omega=eta, n_avrg=100):
    n  = len(w_values)
    ipr_vals = np.zeros(n)
    tmp = np.zeros(n_avrg)
    with Progress(total=n*n_avrg) as p:
        for i in range(n):
            model.set_disorder(w_values[i])
            for j in range(n_avrg):
                p.update()
                tmp[j] = model.inverse_participation_ratio(omega)
            ipr_vals[i] = np.mean(tmp)
    return ipr_vals

def ipr_lengths(model, lengths, omega, w, n_avrg=100):
        n  = len(lengths)
        ipr_vals = np.zeros(n)
        tmp = np.zeros(n_avrg)
        model.set_disorder(w)
        with Progress(total=n*n_avrg) as p:
            for i in range(n):
                model.reshape(lengths[i])
                for j in range(n_avrg):
                    p.update()
                    tmp[j] = model.inverse_participation_ratio(omega)
                ipr_vals[i] = np.mean(tmp)
        return ipr_vals


def plot_ipr(model):
    n = 20
    w_values = np.linspace(0, 10, n)
    ipr_vals = ipr_curve(model, w_values, omega=eta)

    plot = Plot()
    plot.plot(w_values, ipr_vals)
    plot.set_labels(r"$\omega$", "IPR")
    plot.show()


def plot_ipr_length(model):
    n = 20
    lengths = np.arange(5, 50, 5)
    ipr_vals = ipr_lengths(model, lengths, omega=eta, w=2)

    plot = Plot()
    plot.plot(lengths, ipr_vals)
    plot.set_labels(r"L", "IPR")
    plot.show()



def main():
    model = TbDevice.square_p3((1, 5), soc=2)
    plot_ipr_length(model)


if __name__ == "__main__":
    main()
