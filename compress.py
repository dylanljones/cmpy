# -*- coding: utf-8 -*-
"""
Created on  04 2019
author: dylan

project: cmpy
version: 1.0
"""
import sys
import time
import numpy as np
from scipy import linalg as la
from cmpy import *
from cmpy.tightbinding import TbDevice


def time_trans(model, n):
    model.prepare()

    trans = np.zeros(n)
    for i in prange(n):
        trans[i] = model.transmission()
    print(np.mean(trans))


def main():
    model = TbDevice.square_p3((200, 1))
    # model.set_disorder(1)
    time_trans(model, 100)


if __name__ == "__main__":
    main()
