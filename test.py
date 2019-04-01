# -*- coding: utf-8 -*-
"""
Created on 29 Mar 2019
author: Dylan

project: cmpy
version: 1.0
"""
import os
import time
import numpy as np
import psutil
from cmpy import eta, plot_transmission_loss
from cmpy.tightbinding import TbDevice, sp3_basis, s_basis
import pickle

def format_num(num, unit="b", div=1024):
    for scale in ['','k','M','G','T','P','E','Z']:
        if abs(num) < div:
            return f"{num:.1f} {scale}{unit}"
        num /= div

def print_mem():
    vmem = psutil.virtual_memory()
    used = format_num(vmem.used)
    free = format_num(vmem.free)
    total = format_num(vmem.total)
    print(f"Free: {free}, Used: {used}, Total: {total}")

def arr_mem(arr):
    msize = arr.size * arr.itemsize
    print(format_num(msize))


def main():
    b = sp3_basis()
    latt = TbDevice.square((200, 16), eps=b.eps, t=b.hop)
    ham = latt.hamiltonian()
    print_mem()


if __name__ == "__main__":
    main()
