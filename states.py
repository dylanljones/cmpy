#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def dec_to_bin(state, site_num):
    # return str(bin(state))[2:]
    return format(state, f'0{site_num}b')

def bin_to_dec(binary):
    return int(binary, 2)

def electron_count(state):
    return bin(state).count("1")

def occupied(state, site, site_num):
    occ = False
    state_bin = dec_to_bin(state, site_num)
    if state_bin[-site-1]:
        occ = True
    print(occ)
    return occ

def all_states(site_num, particle_num):
    states = []
    for i in range(0, 2**(site_num)):
        # print(dec_to_bin(i))
        if electron_count(i) == particle_num:
            states.append(i)
    return states

def print_states(states, site_num):
    for state in states:
        # site_num = len(dec_to_bin(states[-1]))
        print(dec_to_bin(state, site_num))

def state_to_vector(state, site_num):
    state_bin = dec_to_bin(state, site_num)
    state_vec = [int(d) for d in state_bin]
    return np.array(state_vec)

def vector_to_bin(state_vec):
    return ''.join(str(e) for e in state_vec)

def main():
    site_num = 4
    particle_num = 3
    up_states = all_states(site_num, particle_num)
    dn_states = all_states(site_num, particle_num)
    print_states(up_states, site_num)
    print_states(dn_states, site_num)


    vec1 = state_to_vector(5, 3)
    vec2 = state_to_vector(3, 3)
    kron = np.kron(vec1, vec2)


if __name__ == "__main__":
    main()
