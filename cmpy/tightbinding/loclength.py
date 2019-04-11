# -*- coding: utf-8 -*-
"""
Created on 6 mar 2018
@author: Dylan Jones
project: tightbinding
version: 1.0
"""
import os
from os.path import abspath, dirname, join
import numpy as np
from scipy.optimize import curve_fit
from ..core import eta, Progress

DEFAULT_MODE = "lin"

# =============================================================================
# DATA
# =============================================================================

PROJECT = dirname(dirname(dirname(abspath(__file__))))
DATA_ROOT = join(PROJECT, "_data")

ROOT = join(DATA_ROOT, "localization")
SOC = join(ROOT, "soc")
NO_SOC = join(ROOT, "no soc")
if not os.path.isdir(ROOT):
    os.makedirs(ROOT)


class Folder:

    def __init__(self, path):
        self.root = path
        if not os.path.isdir(path):
            os.makedirs(path)

    def find(self, *txts):
        paths = list()
        if not txts:
            return self.files
        for root, _, files in os.walk(self.root):
            for name in files:
                if all([x in name for x in txts]):
                    paths.append(os.path.join(root, name))
        return paths

    def listdir(self):
        paths = list()
        for name in os.listdir(self.root):
            paths.append(os.path.join(self.root, name))
        return paths

    @property
    def dirs(self):
        dirs = list()
        for path in self.listdir():
            if os.path.isdir(path):
                dirs.append(path)
        return dirs

    @property
    def files(self):
        files = list()
        for path in self.listdir():
            if os.path.isfile(path):
                files.append(path)
        return files


class Data(dict):

    def __init__(self, path=None):
        super().__init__()
        self.path = ""
        if path is not None:
            self.open(path)

    @property
    def filename(self):
        fn = os.path.split(self.path)[1]
        return os.path.splitext(fn)[0]

    @property
    def keylist(self):
        return list(self.keys())

    def save(self):
        np.savez(self.path, **self)

    def open(self, path):
        self.path = path
        if os.path.isfile(self.path):
            self._read()

    def _read(self):
        npzfile = np.load(self.path)
        for key, data in npzfile.items():
            super().update({key: data})


class LT_Data(Data):

    def __init__(self, path):
        super().__init__(path)

    @classmethod
    def new(cls, root, energy, disorder, height):
        filename = f"soc-e={energy}-w={disorder}-h={height}.npz"
        return cls(os.path.join(root, filename))

    @staticmethod
    def key_value(key):
        return float(key.split("=")[1])

    def info(self):
        parts = self.filename.split("-")[1:]
        info = dict()
        for string in parts:
            key, val = str(string).split("=")
            info.update({key: float(val)})
        return info

    def info_str(self):
        return ", ".join([f"{key}={val}" for key, val in self.info().items()])

    def n_avrg(self, key):
        arr = self.get(key, None)
        if arr is None:
            return 0
        else:
            return arr[:, 1:].shape[1]

    def get_set(self, key, mean=True):
        arr = self[key]
        lengths = arr[:, 0]
        trans = arr[:, 1:]
        if mean:
            trans = np.mean(trans, axis=1)
        return lengths, trans

    def sort(self, key):
        arr = self[key]
        idx = np.argsort(arr[:, 0])
        self[key] = arr[idx]

    def sort_all(self):
        for key in self:
            self.sort(key)

    def __str__(self):
        return f"LT-Data({self.info_str()})"

# =============================================================================
# CALCULATIONS
# =============================================================================


def lt_array(model, omega, lengths, n_avrg=100, header=None):
    n = lengths.shape[0]
    arr = np.zeros((n, n_avrg+1))
    sigmas, gammas = model.prepare(omega)
    with Progress(total=n*n_avrg, header=header) as p:
        for i in range(n):
            length = int(lengths[i])
            p.set_description(f"Length={length}")
            model.reshape(length)
            arr[i, 0] = length
            arr[i, 1:] = model.mean_transmission(omega, sigmas, gammas, n_avrg, False, p)
    return arr


def append_trans(model, omega, arr, n_up, header=""):
    lengths = arr[:, 0]
    n = lengths.shape[0]
    new_arr = np.zeros((n, n_up))
    sigmas, gammas = model.prepare(omega)
    with Progress(total=n*n_up, header=header) as p:
        for i in range(n):
            length = int(lengths[i])
            p.set_description(f"Length={length}")
            model.reshape(length)
            new_arr[i] = model.mean_transmission(omega, sigmas, gammas, n_up, False, p)
    return np.append(arr, new_arr, axis=1)


def update_lt(model, omega, existing, n_avrg, new_lengths=None, pre_txt="", post_txt=""):
    arr = existing
    ex_n_avrg = arr[0, 1:].shape[0]
    updated = False

    # Update missing length data-points
    if new_lengths is not None and new_lengths.shape[0]:
        header = pre_txt + "Appending lengths" + post_txt
        new_arr = lt_array(model, omega, new_lengths, ex_n_avrg, header)
        arr = np.append(arr, new_arr, axis=0)
        updated = True

    # Update n_avrg
    n_up = n_avrg - ex_n_avrg
    if n_up > 0:
        header = pre_txt + "Updating avrg-num" + post_txt
        arr = append_trans(model, omega, arr, n_up, header)
        updated = True

    if not updated:
        print(pre_txt + f"Up to date! (n={ex_n_avrg})" + post_txt)
    return arr


def calculate_lt(model, omega, lengths, disorder, n_avrg, existing=None, pre_txt="", post_txt=""):
    model.set_disorder(disorder)
    if existing is None:
        header = pre_txt + "Calculating new" + post_txt
        return lt_array(model, omega, lengths, n_avrg, header)
    else:
        ex_lengths = existing[:, 0]
        new_lengths = np.array([l for l in lengths if l not in ex_lengths])
        arr = update_lt(model, omega, existing, n_avrg, new_lengths, pre_txt, post_txt)
        return arr

# =============================================================================
# INITIALIZATION
# =============================================================================


def lengtharray(l0, l1, n, scale="lin"):
    if scale == "log":
        l0, l1 = np.log10(l0), np.log10(l1)
        lengths = np.logspace(l0, l1, n, dtype="int")
    else:
        lengths = np.linspace(l0, l1, n, dtype="int")
    return np.unique(lengths)


def estimate(model, e, w, lmin=100, lmax=1000, n=40, fitmin=5, fitmax=10, scale="log", n_avrg=200):
    omega = e + eta
    lengths = lengtharray(lmin, lmax, n, scale)
    trans = np.zeros(n)
    loclen, err = 0, 0
    estimation = 0
    length = 0
    model.set_disorder(w)
    sigmas, gammas = model.prepare(omega)
    with Progress(total=n*n_avrg, header="Estimating") as p:
        for i in range(n):
            length = lengths[i]
            info = f"L={length}, Lambda={loclen:.1f}, Err={err:.2f}"
            p.set_description(info)

            # Calculate mean transmission
            model.reshape(length)

            t = np.zeros(n_avrg)
            for j in range(n_avrg):
                p.update()
                t[j] = model.transmission(omega, sigmas=sigmas, gammas=gammas)
            trans[i] = np.log10(np.mean(t))

            # Check localization estimate and error
            if i >= fitmin:
                try:
                    loclen, err = loc_length(lengths[:i], trans[:i], n_fit=fitmax)
                    loclen = abs(loclen)
                    err = abs(err) / loclen
                    if err <= 0.2:
                        estimation = loclen
                        break
                    if 2 * loclen < length:
                        estimation = lmin
                        break
                except Exception as e:
                    print(e)
                    loclen, err = 0, 0
        p.set_description(f"L={length}, Lambda={loclen:.1f}, Err={err:.2f}")

    return int(estimation)


def get_lengths(model, e, w, lmin=100, lmax=1000, n_points=20, step=5, n_avrg=200):
    loclen_est = estimate(model, e, w, lmin, lmax, n_avrg=n_avrg)
    l0 = int(max(loclen_est, 100))
    l1 = l0 + n_points * step
    lengths = np.arange(l0, l1, step)
    return lengths


# =============================================================================
# ANALYSIS
# =============================================================================


def findx(array, x0):
    for i, x in enumerate(array):
        if x >= x0:
            return i
    return 0


def _exp_func(x, l, a):
    return a * np.exp(-x/l)


def _lin_func(x, l, y0):
    return - 1/l * x + y0


def fit(lengths, trans, p0=None, n_fit=None, mode=DEFAULT_MODE):
    if p0 is None:
        p0 = [50, 1]
    if n_fit is not None:
        lengths = lengths[-n_fit:]
        trans = trans[-n_fit:]
    if mode == "lin":
        func = _lin_func
    else:
        func = _exp_func
    popt, pcov = curve_fit(func, lengths, trans, p0=p0)
    return popt, np.sqrt(np.diag(pcov))


def build_fit(lengths, popt, n_fit=None, mode=DEFAULT_MODE):
    lengths = lengths[-n_fit:]
    y = None
    if mode == "exp":
        y = _exp_func(lengths, *popt)
    elif mode == "lin":
        y = _lin_func(lengths, *popt)
    return lengths, y


def loc_length(lengths, trans, p0=None, n_fit=None, mode=DEFAULT_MODE):
    # Fit data, starting from "fit_start"
    popt, errs = fit(lengths, trans, p0, n_fit, mode)
    # return localization-length and -error
    return popt[0], errs[0]


def loc_length_fit(lengths, trans, p0=None, n_fit=None, mode=DEFAULT_MODE):
    # Fit data, starting from "fit_start"
    popt, errs = fit(lengths, trans, p0, n_fit, mode)
    # return localization-length and -error
    return popt[0], errs[0], build_fit(lengths, popt, n_fit, mode)
