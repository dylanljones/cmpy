#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26.11.21

@author: Nico Unglert
"""
from cmpy import matshow
import matplotlib.pyplot as plt
import numpy as np
from cmpy.models import hubbard

fig, ax = plt.subplots()

sites = 6
u = [0.5]*sites
eps = [-1.]*sites
t = 1.
hubb = hubbard.HubbardModel(u=u, eps=eps, t=t)
ham = hubb.hamiltonian(n_up=int(sites/2), n_dn=int(sites/2))

matshow(ham, ax=ax, show=False)
plt.show()