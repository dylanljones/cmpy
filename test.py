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
from cmpy.tightbinding import TbDevice, Folder, LT_Data
from cmpy.tightbinding import disorder_lt, calculate_lt, loc_length
from cmpy.tightbinding.basis import *

TEST_DIR = os.path.join(DATA_DIR, "Tests")
P3_PATH = os.path.join(TEST_DIR, "Localization", "p3-basis")
P3_PATH_2 = os.path.join(TEST_DIR, "Localization", "p3-basis_2")


def conductance(t):
    return t / (1 - t)


def resistance(g):
    return 1 + 1/g


def beta_func(g):
    return -(1 + g) * np.log(resistance(g))


def beta_strong(g, g0):
    return np.log(g / g0)


def check_edge(latt, n_vec, i_dim):
    offset = len(latt.distances)
    x = n_vec[i_dim]
    return x < offset or x > latt.shape[i_dim] - 1 - offset

# =============================================================================

def get_cycle_dist(latt, n1, alpha1, n2, alpha2, d):
    dist = None
    size = latt.shape[d]
    center = size / 2
    if (n1[d] < center) and (n2[d] > center):
        n2_t = n2.copy()
        n2_t[d] -= size-2
        dist = latt.distance((n1, alpha1), (n2_t, alpha2))
    elif (n1[d] > center) and (n2[d] < center):
        n1_t = n1.copy()
        n1_t[d] -= size - 2
        dist = latt.distance((n1_t, alpha1), (n2, alpha2))
    return dist


def get_cycle_neighbours(latt):
    n_dim = len(latt.slice_shape)

    # Find all slice-sites located in proximity of edges
    edges = set()
    for i in range(latt.slice_sites):
        n, alpha = latt.get(i)
        for d in range(n_dim):
            if check_edge(latt, n, d+1):
                edges.add(i)
    edges = list(edges)
    cycle_neighbors = list()
    for i in edges:
        n1, alpha1 = latt.get(i)
        neighours = [list() for _ in range(len(latt.distances))]
        for j in edges:
            if j == i:
                continue
            n2, alpha2 = latt.get(j)
            d = np.where(n1[1:] != n2[1:])[0]
            if len(d):
                d = d[0] + 1
                dist = get_cycle_dist(latt, n1, alpha1, n2, alpha2, d)
                if dist and dist in latt.distances:
                    neighours[latt.distances.index(dist)].append(j)
        cycle_neighbors.append(neighours)
    return cycle_neighbors


def test_cycling():
    model = TbDevice.square_p3((5, 5), soc=0)
    model.set_disorder(0.1)
    ham = model.slice_hamiltonian()
    cycle=False
    if cycle:
        neighbours_cycle = get_cycle_neighbours(model.lattice)
        for i in range(len(neighbours_cycle)):
            neighbors = neighbours_cycle[i]
            print(neighbors)
            for i_dist in range(len(neighbours_cycle[i])):
                dist_neighbours = neighbors[i_dist]
                print(dist_neighbours)
                for j in dist_neighbours:
                    if j > i:
                        t = model.get_hopping(model.lattice.distances[i_dist])
                        ham.set_hopping(i, j, t)

        model._cached_slice = ham

    # model.set_disorder(1)
    lengths = np.arange(50, 100, 10)
    trans = model.transmission_loss(lengths, n_avrg=10, flatten=True)
    plot_transmission_loss(lengths, trans)


# =============================================================================

def plot_scaling():
    model = TbDevice.square((2, 1))
    model.set_disorder(0.5)
    lengths = np.arange(2, 200, 5)

    t0 = model.normal_transmission()
    g0 = conductance(t0)

    t = model.transmission_loss(lengths, n_avrg=200, flatten=True)
    g = conductance(t)

    plot = Plot()
    plot.lines(x=0, y=0, color="0.5", lw=0.5)
    plot.set_limits(xlim=(-4, 4), ylim=(-4, 1))
    x = np.log(g)
    idx = np.argsort(x)
    plot.ax.plot(x[idx], beta_func(g)[idx])
    plot.ax.plot(g, beta_strong(g, g0))
    plot.show()


def main():
    model = TbDevice.square_p3((2, 2))
    ham = model.hamiltonian(1, True)
    ham.show()
    # test_cycling()
    # plot_scaling()



if __name__ == "__main__":
    main()
