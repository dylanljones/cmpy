# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import numpy as np
from bisect import bisect_left
from typing import Callable, Iterable
from .basis import binstr, occupations, overlap
from .operators import (
    project_elements_up,
    project_elements_dn,
)


def project_onsite_energy(up_states, dn_states, eps):
    num_dn = len(dn_states)
    all_up, all_dn = np.arange(len(up_states)), np.arange(num_dn)

    for up_idx, up in enumerate(up_states):
        weights = occupations(up)
        energy = np.sum(eps[:weights.size] * weights)
        yield from project_elements_up(up_idx, num_dn, all_dn, energy)

    for dn_idx, dn in enumerate(dn_states):
        weights = occupations(dn)
        energy = np.sum(eps[:weights.size] * weights)
        yield from project_elements_dn(dn_idx, num_dn, all_up, energy)


def project_interaction(up_states, dn_states, u):
    num_dn = len(dn_states)
    for up_idx, up in enumerate(up_states):
        for dn_idx, dn in enumerate(dn_states):
            weights = overlap(up, dn)
            interaction = np.sum(u[:weights.size] * weights)
            yield from project_elements_up(up_idx, num_dn, dn_idx, interaction)


def _hopping_candidates(num_sites, state, pos):
    results = []
    op = 1 << pos
    occ = state & op

    tmp = state ^ op  # Annihilate or create electron at `pos`
    for pos2 in range(num_sites):
        if pos >= pos2:
            continue
        op2 = (1 << pos2)
        occ2 = state & op2
        # Hopping from `pos` to `pos2` possible
        if occ and not occ2:
            new = tmp ^ op2
            results.append((pos2, new))
        # Hopping from `pos2` to `pos` possible
        elif not occ and occ2:
            new = tmp ^ op2
            results.append((pos2, new))

    return results


def _ordering_phase(state, pos1, pos2=0):
    if pos1 == pos2:
        return 0
    i0, i1 = sorted([pos1, pos2])
    particles = binstr(state)[i0 + 1:i1].count("1")
    return +1 if particles % 2 == 0 else -1


def _compute_hopping(num_sites, states, pos, hopping):
    for i, state in enumerate(states):
        for pos2, new in _hopping_candidates(num_sites, state, pos):
            try:
                t = hopping(pos, pos2)
            except TypeError:
                t = hopping
            if t:
                j = bisect_left(states, new)
                sign = _ordering_phase(state, pos, pos2)
                value = sign * t
                yield i, j, value


def project_site_hopping(up_states, dn_states, num_sites: int,
                         hopping: (Callable, Iterable, float), pos: int):
    num_dn = len(dn_states)
    all_up, all_dn = np.arange(len(up_states)), np.arange(num_dn)

    for up_idx, target, amp in _compute_hopping(num_sites, up_states, pos, hopping):
        yield from project_elements_up(up_idx, num_dn, all_dn, amp, target=target)

    for dn_idx, target, amp in _compute_hopping(num_sites, dn_states, pos, hopping):
        yield from project_elements_dn(dn_idx, num_dn, all_up, amp, target=target)


def project_hopping(up_states, dn_states, num_sites, hopping: (Callable, Iterable, float)):
    for pos in range(num_sites):
        yield from project_site_hopping(up_states, dn_states, num_sites, hopping, pos)
