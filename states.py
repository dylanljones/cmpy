#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cmpy.basis as basis


def dec_to_bin(state_num, site_num):
    # return str(bin(state_num))[2:]
    return format(state_num, f'0{site_num}b')


def bin_to_dec(binary):
    return int(binary, 2)


def electron_count(state_num):
    return bin(state_num).count("1")


def occupied(state_num, site, site_num):
    occ = False
    state_bin = dec_to_bin(state_num, site_num)
    if state_bin[-site - 1]:
        occ = True
    print(occ)
    return occ


def get_all_state_nums(site_num, particle_num):
    states = []
    for i in range(0, 2 ** (site_num)):
        # print(dec_to_bin(i))
        if electron_count(i) == particle_num:
            states.append(i)
    return states


def print_state_bins(states, site_num):
    for state in states:
        # site_num = len(dec_to_bin(states[-1]))
        print(dec_to_bin(state, site_num))


def state_num_to_vector(state_num, site_num):
    state_bin = dec_to_bin(state_num, site_num)
    state_vec = [int(d) for d in state_bin]
    return np.array(state_vec)


def vector_to_bin(state_vec):
    return ''.join(str(e) for e in state_vec)


def get_fullstate_num(up_state_num, dn_state_num, site_num):
    return up_state_num*site_num + dn_state_num

def get_state_num(fullstate_num):
    up_state_num = fullstate_num / 2
    dn_state_num = fullstate_num % 2
    return up_state_num, dn_state_num

def iter_fillings(num_sites):
    for n_up in range(num_sites):
        for n_dn in range(num_sites):
            # print(n_up, n_dn)
            yield n_up, n_dn


def yield_origin(arg1):
    for i in arg1:
        yield i


def yield_step1(arg11):
    arg1 = arg11[:1]
    yield from yield_origin(arg1)


def main():
    for i in yield_step1(np.identity(3)):
        print(i)

    site_num = 4
    particle_num = 3
    up_states = all_states(site_num, particle_num)
    dn_states = all_states(site_num, particle_num)
    print_states(up_states, site_num)

    vec1 = state_to_vector(5, 3)
    vec2 = state_to_vector(3, 3)
    kron = np.kron(vec1, vec2)

    bas = basis.Basis(2)
    print(bas)
    sec = bas.get_sector(1, 1)
    up, dn = sec.up_states, sec.dn_states
    states = list(sec.states)
    print("\nsec.states:")
    for i in states:
        print(i)

    print(sec)
    print(up, dn)

    print(basis.binarr(10))
    iter_fillings(4)


if __name__ == "__main__":
    main()
