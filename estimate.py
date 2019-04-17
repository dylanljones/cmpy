# -*- coding: utf-8 -*-
"""
Created on 11/04/2019
author: dylan

project: cmpy
version: 1.0
"""
import time
import psutil
from threading import Thread
import numpy as np
import matplotlib.pyplot as plt
from cmpy import Symbols, format_num, time_str
from cmpy import eta, ConsoleLine, plot_transmission_loss, Plot

from cmpy.tightbinding import sp3_basis, TbDevice, fit, build_fit, loc_length


class Mean:

    def __init__(self, values=None):
        self.n = 0
        self.delta = 1e10
        self.mean = 0
        if values is not None:
            self.n = len(values)
            self.mean = np.mean(values)

    def reset(self):
        self.n = 0
        self.delta = 1e10
        self.mean = 0

    @property
    def reldelta(self):
        if self.mean:
            return np.abs(self.delta / self.mean)
        else:
            return 1.

    def update(self, x):
        self.n += 1
        self.delta = (x - self.mean) / self.n
        self.mean += self.delta

    def __add__(self, other):
        n = self.n + other.n
        mean = Mean()
        mean.mean = (self.n * self.mean + other.n * other.mean) / n
        mean.n = n
        return mean

    def __str__(self):
        val = np.round(self.mean, 5)
        return f"{val}"


def get_max_size(model, headroom=0.2, lmax=300):
    available = psutil.virtual_memory().available
    for length in range(200, lmax, 20):
        total_orbs = int(model.n_orbs * model.lattice.slice_sites * length)
        size = total_orbs ** 2 * int(np.dtype("complex").itemsize)
        if size >= available:
            return int(100 * np.around((1 - headroom) * length / 100, decimals=1))
    return lmax


def init_estimate_plot(xlim):
    plot = Plot()
    plot.set_limits(xlim, (-5, 1))
    line1 = plot.ax.plot([], [], zorder=2)[0]
    line2 = plot.ax.plot([], [], color="k", zorder=1)[0]
    return plot, line1, line2


def update_plot(plot, line1, line2, trans_data, fit_data):
    ymin = min(plot.ylim[0], 1.2 * min(trans_data[1]))
    line1.set_data(*trans_data)
    line2.set_data(*fit_data)
    plot.set_limits(ylim=(ymin, 1))
    plot.draw()


def loclen_str(loclen, relerr):
    return f"{Symbols.xi}={loclen:.2f}, {Symbols.Delta}{Symbols.xi}={relerr:.2f}"


def timer(func, *args, **kwargs):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        l0 = func(*args, **kwargs)
        print(f"=> L_0={l0}")
        print(f"     t={time_str(time.perf_counter() - t0)}")
        print("----------------------")
        return l0
    return wrapper


@timer
def estimate_navrg(model, lengths, omega=eta, n_batch=10, thresh=0.2):
    n = len(lengths)
    total_batch = n * n_batch
    n_mean = 0

    info_str = "[Batch {0:}] Length {1:}/{2:}: {3:5.1f}%"
    xlim = 0, 1.2 * lengths[-1]
    x_fit = np.linspace(*xlim, 100)

    # plot, line1, line2 = init_estimate_plot(xlim)

    trans = np.zeros(n)
    loclen, err, rel_err = 0, 0, 0

    tmp = np.zeros((n, n_batch))
    sigmas, gammas = model.prepare(omega)

    with ConsoleLine() as out:
        batch = 1
        while True:
            # Calculate one batch of the transmission loss
            i_batch = 1
            for i in range(n):
                model.reshape(lengths[i])
                for j in range(n_batch):
                    string = info_str.format(batch, i+1, n, 100 * i_batch / total_batch)
                    if loclen:
                        string += " -> " + loclen_str(loclen, rel_err)
                    out.write(string)

                    tmp[i, j] = model.transmission(omega, sigmas, gammas)
                    i_batch += 1

            # Update transmission data with new batch
            trans = (n_mean * trans + n_batch * np.mean(tmp, axis=1)) / (n_mean + n_batch)
            n_mean = n_mean + n_batch

            # Check localization length
            log_trans = np.log10(trans)
            popt, errs = fit(lengths, log_trans)
            loclen, err = popt[0], errs[0]
            rel_err = abs(err / loclen)
            # fit_data = build_fit(x_fit, popt)

            # update_plot(plot, line1, line2, (lengths, log_trans), fit_data)
            if rel_err < thresh:
                out.write(f"[Batch {batch}] {Symbols.xi}={loclen:.0f}, {Symbols.Delta}={rel_err:.2f}")
                break
            batch += 1

    return int(np.around(loclen + 2 * err))
    # plot.show()


@timer
def estimate_windowed(model, omega=eta, n=50, n_fit=10, thresh=0.2, mean_thresh=0.5, n_max=2000):
    lengths = np.geomspace(100, 1000, n, endpoint=True, dtype="int")
    trans = [Mean() for _ in range(n)]

    info_str = "[{}/{}] {} Length={}: " + Symbols.Delta + "<T>={:.1f}%"

    sigmas, gammas = model.prepare(omega)
    loclen, err, relerr = 0, 0, 0
    with ConsoleLine() as out:
        for i in range(n):
            model.reshape(lengths[i])

            # Calculate mean-transmission
            counter = 0
            for j in range(n_max):
                string = info_str.format(i+1, n, j, lengths[i], 100*trans[i].reldelta)
                if loclen:
                    string += " -> " + loclen_str(loclen, relerr)
                out.write(string)
                t = model.transmission(omega, sigmas, gammas)
                trans[i].update(t)
                # Check if threshold reached
                if trans[i].reldelta < mean_thresh / 100:
                    counter += 1
                    if counter > 10:
                        break
                elif counter > 0:
                    counter = 0

            # Check localization length
            if i >= n_fit:
                length_window = lengths[i-n_fit:i+1]
                trans_windows = np.log10([t.mean for t in trans[i-n_fit:i+1]])
                popt, errs = fit(length_window, trans_windows)
                loclen, err = popt[0], errs[0]
                relerr = abs(err / loclen)
                # Check if localization-length is much smaller then minimal length
                if 2 * (loclen + err) < lengths[0] and relerr < 0.5:
                    string = f"[{i + 1}/{n}] Length={lengths[i]}: " + loclen_str(loclen, relerr)
                    out.write(string)
                    return lengths[0]
                # Cehck if error is smaller then threshold and window is greater then result
                if relerr < thresh and loclen < length_window[0]:
                    string = f"[{i + 1}/{n}] Length={lengths[i]}: " + loclen_str(loclen, relerr)
                    out.write(string)
                    return int(np.around(loclen + 2 * err))


def estimate_large(model):
    lengths = np.arange(300, 400, 20)
    trans = model.transmission_loss(lengths, n_avrg=100, flatten=True)
    loclen, err = loc_length(lengths, trans)

    print(loclen_str(loclen, err / loclen))


def main():
    model = TbDevice.square_sp3((100, 5))
    model.set_disorder(0.5)

    estimate_large(model)


    # lengths = np.arange(100, 300, 20)
    # print("Batched estimate:")
    # estimate_navrg(model, lengths, n_batch=50)
    #
    # print("Window estimate:")
    # estimate_windowed(model)


if __name__ == "__main__":
    main()
