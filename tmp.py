# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2018
@author: Dylan Jones

project: tmp$
version: 1.0
"""
import numpy as np
from sciutils import Plot


def sus1d(x):
    return 1/x * np.log(np.abs((1+x)/(1-x)))


def sus2d(x):
    res = np.ones_like(x)
    x1 = x[x >= 1]
    res[x >= 1] -= np.sqrt(1-1/(x1*x1))
    return res


def sus3d(x):
    return 1 + (1-x*x)/(2*x) * np.log(np.abs((1+x)/(1-x)))


def e_sc(x, delta=1, sign=1):
    return x * sign * np.sqrt((x*x/2) + np.abs(delta)**2)


def main():
    x = np.linspace(-10, 10, 1000)
    plot = Plot()
    plot.plot(x, e_sc(x, 1, 1), label="1D")
    plot.show()


if __name__ == "__main__":
    main()
