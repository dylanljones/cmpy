# -*- coding: utf-8 -*-
"""
Created on 31 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
import numpy as np
from scipy import random
import matplotlib.pyplot as plt
from cmpy import Plot, prange


def f(x):
    return np.sin(x)


def monte_carlo_int(func, a=0, b=np.pi, n=1000000):
    x_rand = random.uniform(a, b, n)
    integral = 0
    for i in prange(n):
        integral += func(x_rand[i])
    return (b-a)/n * integral


def main():
    print(monte_carlo_int(f))

if __name__ == "__main__":
    main()
