# -*- coding: utf-8 -*-
"""
Created on 26 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
import re
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from cmpy import *
from cmpy.tightbinding import *
from cmpy.tightbinding.basis import *

TEST_DIR = os.path.join(DATA_DIR, "Tests")
create_dir(TEST_DIR)


def sort_files():
    new_root = os.path.join(TEST_DIR, "Localization", "p3-basis")
    for fn in os.listdir(PROJECT_DIR):
        if fn.endswith(".npz"):
            path = os.path.join(PROJECT_DIR, fn)
            soc = int(search_string_value(fn, "soc="))
            #print(soc, h)
            new_dir = os.path.join(new_root, f"soc={soc}")
            create_dir(new_dir)

            new_path = os.path.join(new_dir, fn)
            shutil.copyfile(path, new_path)



def test_logarithmic(w_values, h, soc, n_avrg=100):
    model = TbDevice.square_p3((200, h), soc=soc)
    #lengths = np.arange(100, 201, 10)
    root = os.path.join(TEST_DIR,  "Localization", "p3-basis", f"soc={soc}")
    create_dir(root)
    path = os.path.join(root, f"test-h={h}-soc={soc}.npz")
    data = LT_Data(path)
    for w in w_values:
        model.set_disorder(w)

        lengths = get_lengths(model, eta, w, lmin=50)
        trans = model.transmission_loss(lengths, n_avrg=n_avrg)
        arr = np.zeros((len(lengths), n_avrg+1))
        arr[:, 0] = lengths
        arr[:, 1:] = trans
        data.update({f"w={w}": arr})
        data.save()


def create_test_data(heights, soc):
    w = 1, 2, 4, 8, 16
    for h in heights:
        print(f"h={h}")
        test_logarithmic(w, h, soc)


def sort_paths(paths, query="h="):
    heights = [int(re.search(query + "(\d+)", p).group(1)) for p in paths]
    idx = np.argsort(heights)
    return [paths[i] for i in idx]

def search_string_value(string, header):
    return re.search(header + "(\d+)", string).group(1)


def show_loclen(*socs):
    root = os.path.join(TEST_DIR, "Localization", "p3-basis")
    for dirname in os.listdir(root):
        dirpath = os.path.join(root, dirname)
        if len(socs) and not any([f"soc={s}" in dirname for s in socs]):
            continue

        data_list = list()
        for path in list_files(dirpath):
            data = LT_Data(path)

            h = data.info()["h"]
            w, ll = list(), list()
            for k in data:
                l, t = data.get_set(k, mean=True)
                w.append(data.key_value(k))
                ll.append(loc_length(l, np.log10(t))[0])
            data_list.append((h, w, ll))

        plot = Plot()
        plot.set_title(dirname)
        for h, w, ll in sorted(data_list, key=lambda x: x[0]):
            plot.plot(w, np.log10(ll), label=f"M={h}")
        plot.legend()
    plot.show()






def main():
    model = TbDevice.square_p3((200, 1), eps_p=0, soc=1)

    #create_test_data([1, 4, 8, 16], 3)
    #create_test_data([1, 4, 8, 16], 4)
    show_loclen(1, 2, 3)




if __name__ == "__main__":
    main()
