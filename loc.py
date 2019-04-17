# -*- coding: utf-8 -*-
"""
Created on 3 Mar 2018
@author: Dylan Jones

project: tightbinding
version: 1.0
"""
import os
import numpy as np
from cmpy import eta
from cmpy.tightbinding import TbDevice, s_basis, p3_basis, sp3_basis
from cmpy.tightbinding.loclength import LT_Data, Folder, S_PATH, P3_PATH, SP3_PATH
from cmpy.tightbinding.loclength import calculate_lt, get_lengths


def print_header(txt):
    print()
    print("-" * len(txt))
    print(txt)
    print("-" * len(txt))


def init_lengths(lengths, existing, model, e, w, n_avrg=200):
    # if lengths is None:
    #     if existing is None:
    #         if w <= 2:
    #             print("Initializing lengths")
    #             out = get_lengths(model, e, w, n_avrg=n_avrg)
    #             print(f"Configured range: {out[0]}-{out[-1]}")
    #         else:
    #             out = np.arange(100, 200, 5)
    #     else:
    #         out = existing[:, 0]
    # else:
    #     out = lengths
    # return out
    return np.arange(200, 300, 5)


def calculate_width_lt(basis, heights, w, lengths=None, e=0, n_avrg=250):
    omega = e + eta
    # initialize data
    if basis.n == 1:
        fn = f"width-e={e}-w={w}.npz"
        data = LT_Data(os.path.join(S_PATH, fn))
    else:
        fn = f"width-e={e}-w={w}-soc={basis.soc}.npz"
        if basis.n == 6:
            data = LT_Data(os.path.join(P3_PATH, fn))
        elif basis.n == 8:
            data = LT_Data(os.path.join(SP3_PATH, fn))

    header = f"Calculating L-T-Dataset (e={e}, w={w}, soc={basis.soc})"
    print_header(header)
    n = len(heights)
    for i in range(n):
        h = heights[i]
        key = f"h={h}"
        print(f"{i+1}/{n} (height: {h}) ")
        model = TbDevice.square((2, h), basis.eps, basis.hop)

        existing = data.get(key, None)
        sys_lengths = init_lengths(lengths, existing, model, e, w)

        arr = calculate_lt(model, sys_lengths, w, n_avrg, omega, existing=existing)
        data.update({key: arr})
        data.save()
        print()
    print()
    return data


def calculate_disorder_lt(basis, w_values, h, lengths=None, e=0, n_avrg=250):
    omega = e + eta
    # initialize data
    if basis.n == 1:
        fn = f"disord-e={e}-h={h}.npz"
        data = LT_Data(os.path.join(S_PATH, fn))
    else:
        fn = f"disord-e={e}-h={h}-soc={basis.soc}.npz"
        if basis.n == 6:
            data = LT_Data(os.path.join(P3_PATH, fn))
        elif basis.n == 8:
            data = LT_Data(os.path.join(SP3_PATH, fn))

    # calculate dataset
    header = f"Calculating L-T-Dataset (e={e}, h={h}, soc={basis.soc})"
    print_header(header)
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
        print()
    return data


def mean_batched(basis, w_values, heights, n_batch=500, n_avrg=3000):
    for n in range(n_batch, n_avrg, n_batch):
        for h in heights:
            calculate_disorder_lt(basis, w_values, h, n_avrg=n)


def new_lengths(lengths, offset):
    step = lengths[1] - lengths[0]
    l0 = int(lengths[-1] + step)
    l1 = l0 + offset
    new = np.arange(l0, l1, step)
    return new


def update_lengths(data, offset=50):
    name = os.path.split(os.path.dirname(data.path))[1]
    info = data.info()

    if name == "s-basis":
        basis = s_basis()
    elif name == "p3-basis":
        soc = info["soc"]
        basis = p3_basis(soc=soc)
    else:
        soc = info["soc"]
        basis = sp3_basis(soc=soc)

    h = int(info["h"])
    e = info["e"]
    omega = e + eta

    header = f"Calculating L-T-Dataset (e={e}, h={h}, soc={basis.soc})"
    print_header(header)
    model = TbDevice.square((2, h), basis.eps, basis.hop)
    n = len(data.keylist)
    for i, key in enumerate(data):
        w = data.key_value(key)
        print(f"{i+1}/{n} (disorder: {w}) ")
        existing = data.get(key, None)
        lengths = new_lengths(existing[:, 0], offset)
        n_avrg = len(existing[0, 1:])

        arr = calculate_lt(model, lengths, w, n_avrg, omega, existing=existing)
        data.update({key: arr})
        # data.sort_all()
        data.save()
        print()


def update_all_lengths(root, l_offset):
    folder = Folder(root)
    for path in folder.files:
        data = LT_Data(path)
        update_lengths(data, l_offset)
        break


def main():
    # basis = s_basis()
    basis = sp3_basis(soc=1)

    w_values = [1, 1.5, 2, 2.5, 3, 4, 5]
    heights = [1, 2, 4, 6]
    # update_all_lengths(S_PATH, 10)

    mean_batched(basis, w_values, heights, n_avrg=2000, n_batch=500)

    #for h in heights:
    #    calculate_disorder_lt(basis, w_values, h, n_avrg=1000)


if __name__ == "__main__":
    main()
