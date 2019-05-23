# -*- coding: utf-8 -*-
"""
Created on 6 mar 2018
@author: Dylan Jones
project: tightbinding
version: 1.0
"""
import os
import numpy as np
from scipy.optimize import curve_fit
from ..core import eta
from ..core import Progress, ConsoleLine, Symbols
from ..core import Folder, Data
from .basis import s_basis, p3_basis, sp3_basis
from .device import TbDevice

DEFAULT_MODE = "lin"

# =============================================================================
# DATA
# =============================================================================


class LT_Data(Data):

    def __init__(self, path):
        super().__init__(path)

    @staticmethod
    def key_value(key):
        return float(key.split("=")[1])

    def info(self):
        parts = self.filename.split("-")[1:]
        info = dict()
        for string in parts:
            key, val = str(string).split("=")
            if val != "None":
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

    def get_set(self, key, mean=False):
        arr = self[key]
        lengths, trans = arr[:, 0], arr[:, 1:]
        if mean:
            trans = np.mean(trans, axis=1)
        return lengths, trans

    def get_lengths(self, key):
        arr = self[key]
        return arr[:, 0]

    def get_trans(self, key, mean=True):
        arr = self[key]
        return np.mean(arr[:, 1:], axis=1) if mean else arr[:, 1:]

    def sort_lengths(self, key):
        arr = self[key]
        idx = np.argsort(arr[:, 0])
        self[key] = arr[idx]

    def sort_all(self):
        for key in self:
            self.sort_lengths(key)

    def get_disord_model(self, key):
        basis_name = os.path.split(os.path.dirname(self.path))[1]
        info = self.info()
        if basis_name == "s-basis":
            basis = s_basis()
        elif basis_name == "p3-basis":
            soc = info["soc"]
            basis = p3_basis(soc=soc)
        else:
            soc = info["soc"]
            basis = sp3_basis(soc=soc)

        e = info["e"]
        h = int(self.key_value(key))
        model = TbDevice.square((2, h), basis.eps, basis.hop)
        return model, e + eta

    def __str__(self):
        return f"LT-Data({self.info_str()})"

# =============================================================================
# INITIALIZATION
# =============================================================================


def _lengtharray(l0, l1, n, scale="lin"):
    if scale == "log":
        l0, l1 = np.log10(l0), np.log10(l1)
        lengths = np.logspace(l0, l1, n, dtype="int")
    else:
        lengths = np.linspace(l0, l1, n, dtype="int")
    return np.unique(lengths)


def estimate(model, e, w, lmin=100, lmax=1000, n=40, fitmin=5, fitmax=10, scale="log", n_avrg=200,
             header=None):
    if w == 0:
        raise ValueError("Can't estimate localization-length without any disorder")

    omega = e + eta
    lengths = _lengtharray(lmin, lmax, n, scale)
    trans = np.zeros(n)
    loclen, err, relerr = 0, 0, 0
    estimation = 0
    length = 0
    model.set_disorder(w)
    sigmas, gammas = model.prepare(omega)
    if header is None:
        header = f"Estimating"
    with ConsoleLine(header=header) as out:
        for i in range(n):
            length = lengths[i]
            info = f"L={length}, Loclen={loclen:.2f}, Err={relerr:.2f}"

            # Calculate mean transmission
            model.reshape(length)
            t = np.zeros(n_avrg)
            for j in range(n_avrg):
                # p.update()
                p = f"({i+1}/{n}) {100 * j / n_avrg:5.1f}% "
                out.write(p + info)
                t[j] = model.transmission(omega, sigmas=sigmas, gammas=gammas)
            trans[i] = np.mean(np.log(t))

            # Check localization estimate and error
            if i >= fitmin:
                try:
                    loclen, err = loc_length(lengths[:i], trans[:i], n_fit=fitmax)
                    loclen = abs(loclen)
                    relerr = abs(err) / loclen
                    if relerr <= 0.2:
                        estimation = loclen
                        break
                    if relerr <= 1 and 2 * loclen < length:
                        estimation = lmin
                        break
                except Exception as e:
                    print(e)
                    loclen, err, relerr = 0, 0, 0

        out.write(f"({i+1}/{n}) L={length}, Loclen={loclen:.2f}, Err={relerr:.2f}")
    return int(estimation)


def get_lengths(model, e, w, lmin=100, lmax=1000, n_points=20, step=5, n_avrg=200):
    loclen_est = estimate(model, e, w, lmin, lmax, n_avrg=n_avrg)
    l0 = int(max(loclen_est, lmin))
    l1 = l0 + n_points * step
    lengths = np.arange(l0, l1, step)
    return lengths

# =============================================================================
# CALCULATIONS
# =============================================================================


def _lt_array(model, omega, lengths, n_avrg=100, header=None):
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


def _append_trans(model, omega, arr, n_up, header=""):
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


def _update_lt(model, omega, existing, n_avrg, new_lengths=None, pre_txt="", post_txt=""):
    arr = existing
    ex_n_avrg = arr[0, 1:].shape[0]
    updated = False

    # Update missing length data-points
    if new_lengths is not None and new_lengths.shape[0]:
        header = pre_txt + "Appending lengths" + post_txt
        new_arr = _lt_array(model, omega, new_lengths, ex_n_avrg, header)
        arr = np.append(arr, new_arr, axis=0)
        updated = True

    # Update n_avrg
    n_up = n_avrg - ex_n_avrg
    if n_up > 0:
        header = pre_txt + "Updating avrg-num" + post_txt
        arr = _append_trans(model, omega, arr, n_up, header)
        updated = True

    if not updated:
        print(pre_txt + f"Up to date! (n={ex_n_avrg})" + post_txt)
    return arr


def calculate_lt(model, lengths, disorder, n_avrg, omega=eta, existing=None, pre_txt="", post_txt=""):
    model.set_disorder(disorder)
    if existing is None:
        header = pre_txt + "Calculating new" + post_txt
        return _lt_array(model, omega, lengths, n_avrg, header)
    else:
        ex_lengths = existing[:, 0]
        new_lengths = np.array([l for l in lengths if l not in ex_lengths])
        arr = _update_lt(model, omega, existing, n_avrg, new_lengths, pre_txt, post_txt)
        return arr


def _print_header(txt):
    print()
    print("-" * len(txt))
    print(txt)
    print("-" * len(txt))


def init_lengths(lengths, existing, model, e, w, lmin=50, lmax=2000, n=20, step=5, n_avrg=200):
    if lengths is None:
        if existing is None:
            loclen_est = estimate(model, e, w, lmin, lmax, n_avrg=n_avrg)
            l0 = int(max(loclen_est, lmin))
            l1 = l0 + n * step
            out = np.arange(l0, l1, step)
            print(f"-> Configured range: {out[0]}-{out[-1]}")
        else:
            out = existing[:, 0]
    else:
        out = lengths
    return out


def disorder_lt(path, basis, h, w_values, lengths=None, e=0, n_avrg=250):
    omega = e + eta
    # initialize data
    data = LT_Data(path)

    # calculate dataset
    header = f"Calculating L-T-Dataset (e={e}, h={h}, soc={basis.soc}, n={n_avrg})"
    _print_header(header)
    model = TbDevice.square((2, h), basis.eps, basis.hop)

    n = len(w_values)
    for i in range(n):
        w = w_values[i]
        key = f"w={w}"
        print(f"{i+1}/{n} (disorder: {w}) ")

        existing = data.get(key, None)
        sys_lengths = init_lengths(lengths, existing, model, e, w, n_avrg=500)

        arr = calculate_lt(model, sys_lengths, w, n_avrg, omega, existing=existing)
        data.update({key: arr})
        data.save()
        #print()
    print()
    return data

# =============================================================================
# ANALYSIS
# =============================================================================


def findx(array, x0):
    for i, x in enumerate(array):
        if x >= x0:
            return i
    return 0


def _exp_func(x, l, a):
    return a * np.exp(-2*x/l)


def _lin_func(x, l, y0):
    return - (2 * x / l) + y0


def fit(lengths, trans, p0=None, n_fit=None, mode=DEFAULT_MODE):
    if p0 is None:
        p0 = [1, 1]
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
    if n_fit is not None:
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
