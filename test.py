# -*- coding: utf-8 -*-
"""
Created on 26 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
import time
import numpy as np
from threading import Thread
from scipy import linalg as la
from scipy.integrate import quad
import matplotlib.pyplot as plt
from cmpy import Progress, prange, ConsoleLine, eta
from cmpy.tightbinding import TbDevice, sp3_basis, loc_length


class MeanThread(Thread):

    INDEX = 0

    def __init__(self, shape, basis, disorder, omega=eta, n=1000):
        self.idx = MeanThread.INDEX
        MeanThread.INDEX += 1
        self.model = TbDevice.square(shape, basis.eps, basis.hop)
        self.model.set_disorder(disorder)

        self.i, self.n = 0, n
        self.omega = omega
        self.array = np.zeros(n)

        super().__init__()

    @property
    def finished(self):
        return self.i == self.n - 1

    def run(self):
        sigmas, gammas = self.model.prepare(self.omega)
        for i in range(self.n):
            self.i = i
            self.array[i] = self.model.transmission(self.omega, sigmas, gammas)


def calculate(basis, lengths, disorder, n_avrg):
    n = len(lengths)
    with ConsoleLine() as out:
        threads = list()
        for l in lengths:
            t = MeanThread((l, 1), basis, disorder, n=n_avrg)
            threads.append(t)
            out.write(f"Starting Thread {t.idx}")
            t.start()
            
        done = False
        while not done:
            time.sleep(1)
            indices = [t.i + 1 for t in threads]
            prog = sum(indices) / (n * n_avrg)
            done = prog == 1.
            out.write(f"Progress: {100*prog:.1f}%q")

    for t in threads:
        t.join()
    trans = list()
    for t in threads:
        trans.append(t.array)
    return np.array(trans)


def main():
    basis = sp3_basis()
    lengths = np.arange(100, 200, 5)
    disorder = 1
    n = 1000
    trans = calculate(basis, lengths, disorder, n)

    plt.plot(lengths, np.log10(np.mean(trans, axis=1)))
    plt.show()


if __name__ == "__main__":
    main()
