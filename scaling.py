# -*- coding: utf-8 -*-
"""
Created on 15 Apr 2019
@author: Dylan Jones

project: tightbinding
version: 1.0
"""
import matplotlib.pyplot as plt
from cmpy import Plot
from cmpy.tightbinding.loclength import *

folder = Folder(SP3_PATH)


def beta(trans, t0):
    return np.log(trans / t0)



def main():
    data = LT_Data(folder.files[1])

    plot = Plot()
    print(data.info_str())
    for key in data.keys():
        model, omega = data.get_disord_model(key)
        t0 = model.normal_transmission(omega)
        l, t = data.get_set(key, mean=True)
        scaling = beta(t, t0)
        plot.plot(t, scaling, label=key)
    plot.legend()
    plot.show()



if __name__ == "__main__":
    main()
