# -*- coding: utf-8 -*-
"""
Created on  06 2019
author: dylan

project: cmpy2
version: 1.0
"""
import numpy as np
from sciutils import fit, build_fit, eta, Data
from sciutils import Progress, Plot


def transfunc(x, l, y0):
    return - (2 / l) * x + y0


def localization_length(x, y, l0=100):
    popt, errs = fit(transfunc, x, y, [l0, 0])
    return popt[0], errs[0]


def localization_length_built(x, y, l0=100):
    popt, errs = fit(transfunc, x, y, [l0, 0])
    loclen = popt[0], errs[0]
    data = build_fit(transfunc, x, popt)
    return loclen, data


# =============================================================================
# DATA
# =============================================================================


class LT_Data(Data):

    def __init__(self, *paths):
        path = os.path.join(*paths)
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


def estimate(model, e, w, lmin=100, lmax=1000, n=40, fitmin=10, fitmax=15, thresh=0.1, scale="log",
             n_avrg=200):
    if w == 0:
        raise ValueError("Can't estimate localization-length without any disorder")

    omega = e + eta
    if scale == "log":
        lmin, lmax = np.log10(lmin), np.log10(lmax)
        lengths = np.logspace(lmin, lmax, n, dtype="int")
    else:
        lengths = np.linspace(lmin, lmax, n, dtype="int")
    # Remove double lengths (They sometimes occure due to the conversion to int)
    lengths = np.unique(lengths)

    # Set up arrays, constants and the model
    trans = np.zeros(n)
    loclen, err, relerr = 0, 0, 0
    estimation = 0
    length = 0
    model.set_disorder(w)

    with Progress(header="Estimating") as out:
        # Start estimation loop
        for i in range(n):
            # Set model length
            length = lengths[i]
            model.reshape(length)
            info = f"L={length}, Loclen={loclen:.2f}, Err={relerr:.2f}"

            # Calculate mean logarithmic transmission
            t = np.zeros(n_avrg)
            for j in range(n_avrg):
                p = f"({i+1}/{n}) {100 * j / n_avrg:5.1f}% "
                out.update(txt=p + info)
                t[j] = model.transmission(omega)
                model.shuffle()
            trans[i] = np.mean(np.log(t))

            # Check localization estimate and error
            if i >= fitmin:
                try:
                    # Get the last fitmax datapoints
                    i0, i1 = max(0, i-fitmax), i
                    x = lengths[i0:i1]
                    y = trans[i0:i1]
                    # Fit section and get loc-length
                    loclen, err = localization_length(x, y)
                    loclen = abs(loclen)
                    # Check relative error of loc-length and continue if too big
                    relerr = abs(err) / loclen
                    if relerr <= thresh:
                        estimation = loclen
                        break
                    if relerr <= 1 and 2 * loclen < length:
                        estimation = lmin
                        break
                except Exception as e:
                    # Catch any fitting-errors and prevent loop from stoping
                    print(e)
                    loclen, err, relerr = 0, 0, 0
                    
    return int(estimation)


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


# =============================================================================
# CALCULATIONS
# =============================================================================


def _lt_array(model, omega, lengths, n_avrg=100, header=None):
    n = lengths.shape[0]
    arr = np.zeros((n, n_avrg+1))
    with Progress(total=n*n_avrg, header=header) as p:
        for i in range(n):
            length = int(lengths[i])
            p.set_description(f"Length={length}")
            model.reshape(length)
            arr[i, 0] = length
            arr[i, 1:] = model.transmission_mean(omega, n=n_avrg, flatten=False, prog=p)
    return arr


def _append_trans(model, omega, arr, n_up, header=""):
    lengths = arr[:, 0]
    n = lengths.shape[0]
    new_arr = np.zeros((n, n_up))
    with Progress(total=n*n_up, header=header) as p:
        for i in range(n):
            length = int(lengths[i])
            p.set_description(f"Length={length}")
            model.reshape(length)
            new_arr[i] = model.transmission_mean(omega, n=n_up, flatten=False, prog=p)
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


# def disorder_lt(path, basis, h, w_values, lengths=None, e=0, n_avrg=250):
#     omega = e + eta
#     # initialize data
#     data = LT_Data(path)
#
#     # calculate dataset
#     header = f"Calculating L-T-Dataset (e={e}, h={h}, soc={basis.soc}, n={n_avrg})"
#     line = "-" * len(header)
#     print("\n" + line + "\n" + header + "\n" + line)
#
#     model = TbDevice.square((2, h), basis.eps, basis.hop)
#     n = len(w_values)
#     for i in range(n):
#         w = w_values[i]
#         key = f"w={w}"
#         print(f"{i+1}/{n} (disorder: {w}) ")
#         existing = data.get(key, None)
#
#         sys_lengths = init_lengths(lengths, existing, model, e, w, n_avrg=500)
#         arr = calculate_lt(model, sys_lengths, w, n_avrg, omega, existing=existing)
#         data.update({key: arr})
#         data.save()
#
#     print()
#     return data
