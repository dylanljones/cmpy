#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
import numpy as np

M = 4
# for ii in range(2**M):
#     print(f"state number {ii:3}: {ii:0{M}b}")
# print(4**M, 16*16)

state = 9


def occ_total(state):
    return bin(state).count("1")


# checking a specific site:
# for pos in range(M):
#     print(f"Electron at site {pos}: {bool(state & 1 << pos)}")

def demo_np_slice():
    print(
        bin(6), "\n",
        bin(6)[::-1], "\n",
        bin(6)[:0:-1], "\n",
        bin(6)[:1:-1], "\n",
        bin(6)[:2:-1], "\n",
        bin(6)[:3:-1], "\n"
    )


def demo_xor():
    num_spin = 0b1101
    print(
        "IMPURITY = ", IMPURITY, bin(IMPURITY)[2:], "\n",
        "num_spin = ", num_spin, bin(num_spin)[2:], "\n",
        "nump_spin ^ IMPURITY = ", num_spin ^ IMPURITY, bin(num_spin ^ IMPURITY)[2:]
    )


def create_spinstates(size: int, return_state=False):
    spinstates = defaultdict(list)
    max_int = int('1' * size, base=2)

    for state in range(max_int + 1):
        spinstates[occ_total(state)].append(state)
    # spinstates = dict(spinstates)
    for key, val in spinstates.items():
        spinstates[key] = np.array(val)
    return spinstates


print(create_spinstates(3))

states = create_spinstates(3)

example_state = states[2][2]
e_onsite = np.array([1, 2, 3, 4])

# print(example_state, bin(example_state)[2:], bin(example_state)[:1:-1])
# # on-site energies
# bit_array = np.array([int(ss) for ss in bin(example_state)[:1:-1]], dtype=np.bool_)
# print(bit_array.astype(np.int8))
# print(e_onsite[:bit_array.size] * bit_array)
# print(np.sum(e_onsite[:bit_array.size] * bit_array))

# interaction
# up_state = states[2][1]
# dn_state = states[1][0]
# print(f"up {up_state}", bin(up_state)[2:])
# print(f"dn {dn_state}", bin(dn_state)[2:])
# U = 7
IMPURITY = 0b1
#
# overlap = IMPURITY & up_state & dn_state
# print(overlap * U)


def _calc_hopp_from_center(num_spin: int):
    """How you would implement it in python instead."""
    # c_0
    if not num_spin & IMPURITY:  # no electron that can hop
        return
    num_new = num_spin ^ IMPURITY  # remove center electron

    sign = +1  # every time we hop past an electron we pick up a - sign
    for kk, vk in enumerate(V, start=1):
        if num_spin & (1 << kk):  # state filled, no hopping but sign change
            sign *= -1
        else:
            yield sign * vk, num_new ^ (1 << kk)


def hopping(V, num_spin, target_site):
    if not num_spin & IMPURITY or num_spin & (1 << target_site):  # no electron that can hop
        return
    num_new = num_spin ^ IMPURITY  # remove center electron
    sign = +1
    for kk in range(1, target_site+1):
        if num_spin & (1 << kk):
            sign *= -1
    return sign * V[kk], num_new ^ (1 << kk)


def main():
    # demo_np_slice()
    # demo_xor()
    V = [1, 2, 3, 4]
    num_spin = 5
    site_num = 4
    print(bin(num_spin)[2:])
    a = hopping(V, num_spin, 1)
    print(a, bin(a[1])[2:])


if __name__ == "__main__":
    main()
