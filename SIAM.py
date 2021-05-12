#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def get_all_states(particle_num):
    states = []
    for i in range(0, 2 ** (site_num)):
        if electron_count(i) == particle_num:
            states.append(i)
    return states


def dec_to_bin(state):
    return bin(state)[2:]


def bin_to_dec(binary):
    return int(binary, 2)


def electron_count(state):
    return bin(state).count("1")


def occupied(state, site):
    return bool(state & (1 << site))


def state_to_vector(state):
    state_vec = [int(d) for d in dec_to_bin(state)]
    return np.array(state_vec, dtype=np.bool_)


def vector_to_bin(state_vec):
    return ''.join(str(e) for e in state_vec)


def vector_to_state(vector):
    state_bin = vector_to_bin(vector)
    state_vec = bin_to_dec(state_bin)
    return state_vec


def get_state_num(fullstate_num):
    up_state_num = fullstate_num / 2
    dn_state_num = fullstate_num % 2
    return up_state_num, dn_state_num


def get_fullstate_num(up_state_num, dn_state_num):
    return up_state_num * site_num + dn_state_num


def compute_onsite(state):
    rev_bit_array = state_to_vector(state)[::-1]
    return np.sum(eps_onsite[:rev_bit_array.size] * rev_bit_array)


def compute_interaction(up_state, dn_state, U):
    overlap = IMPURITY & up_state & dn_state
    return overlap * U


def compute_hopping_out(V, num_spin, target_site):
    if not num_spin & IMPURITY or num_spin & (1 << target_site):  # no electron that can hop
        return
    num_new = num_spin ^ IMPURITY  # remove center electron
    sign = +1
    for kk in range(1, target_site):
        if num_spin & (1 << kk):
            sign *= -1
    return sign * V[target_site], num_new ^ (1 << target_site)


def get_hopping_out_el(up, dn, full, up_idx, dn_idx):
    hopp_up_elements = []
    hopp_dn_elements = []
    for site in range(1, site_num):
        larry = compute_hopping_out(V, up, site)
        if larry is not None:
            hopp_up_e, new_up_state = larry
            full_hopp_up = get_fullstate_num(up_states.index(new_up_state), dn_idx)
            hopp_up_elements.append([full_hopp_up, full, hopp_up_e])

        barry = compute_hopping_out(V, dn, site)
        if barry is not None:
            hopp_dn_e, new_dn_state = barry
            full_hopp_dn = get_fullstate_num(up_idx, dn_states.index(new_dn_state))
            hopp_dn_elements.append([full_hopp_dn, full, hopp_dn_e])

    return hopp_dn_elements + hopp_up_elements

def compute_hopping_in(V, num_spin, target_site):
    if num_spin & IMPURITY or not num_spin & (1 << target_site):  # no electron that can hop
        return
    num_new = num_spin ^ (1 << target_site)  # remove target electron
    sign = +1
    for kk in range(1, target_site):
        if num_spin & (1 << kk):
            sign *= -1
    return sign * V[target_site], num_new + 1


def get_hopping_in_el(up, dn, full, up_idx, dn_idx):
    hopp_up_elements = []
    hopp_dn_elements = []
    for site in range(1, site_num):
        larry = compute_hopping_in(V, up, site)
        if larry is not None:
            hopp_up_e, new_up_state = larry
            full_hopp_up = get_fullstate_num(up_states.index(new_up_state), dn_idx)
            hopp_up_elements.append([full_hopp_up, full, hopp_up_e])

        barry = compute_hopping_in(V, dn, site)
        if barry is not None:
            hopp_dn_e, new_dn_state = barry
            full_hopp_dn = get_fullstate_num(up_idx, dn_states.index(new_dn_state))
            hopp_dn_elements.append([full_hopp_dn, full, hopp_dn_e])

    return hopp_dn_elements + hopp_up_elements


def get_onsite_el(up, dn, full):
    e_up = compute_onsite(up)
    e_dn = compute_onsite(dn)
    return [full, full, e_up + e_dn]


def get_inter_el(up, dn, full):
    e_inter = compute_interaction(up, dn, U)
    return [full, full, e_inter]


def get_matrix_elements(up_states, dn_states):
    onsite_elements = []
    interaction_elements = []
    hopping_elements = []
    for up_idx, up in enumerate(up_states):
        for dn_idx, dn in enumerate(dn_states):
            full = get_fullstate_num(up_idx, dn_idx)

            onsite_elements.append(get_onsite_el(up, dn, full))

            interaction_elements.append(get_inter_el(up, dn, full))

            hopping_elements += get_hopping_out_el(up, dn, full, up_idx, dn_idx)
            hopping_elements += get_hopping_in_el(up, dn, full, up_idx, dn_idx)
    return onsite_elements + interaction_elements + hopping_elements

def set_hamiltonian(matrix_elements, up_states, dn_states):
    dim = len(up_states) * len(dn_states)
    from scipy.sparse import csr_matrix
    col = matrix_elements[:, 0]
    row = matrix_elements[:, 1]
    data = matrix_elements[:, 2]
    hamiltonian = csr_matrix((data, (row, col)), shape=(dim, dim)).toarray()
    print(np.shape(hamiltonian))
    return hamiltonian


IMPURITY = 0b1
site_num = 3
eps_onsite = [1, 2, 3, 4]
U = 4
V = [1, 2, 3, 4]


# ========================================================================
n_up = 3
n_dn = 3
up_states = get_all_states(n_up)
dn_states = get_all_states(n_dn)
matel = np.array(get_matrix_elements(up_states, dn_states))
hamiltonian = set_hamiltonian(matel, up_states, dn_states)
print(hamiltonian)

# ========================================================================


# gs = None
# e_gs = np.infty
# gs_sec = None
#
# for n_up in range(0, site_num+1):
#     for n_dn in range(0, site_num+1):
#         print(f"------- n_up {n_up}, n_dn {n_dn}")
#         up_states = get_all_states(n_up)
#         dn_states = get_all_states(n_dn)
#         matel = np.array(get_matrix_elements(up_states, dn_states))
#         # print(matel)
#         hamiltonian = set_hamiltonian(matel, up_states, dn_states)
#         print(hamiltonian)
#         eigvals, eigvecs = np.linalg.eigh(hamiltonian)
#         i0 = np.argmin(eigvals)
#         e0 = eigvals[i0]
#         if e0 < e_gs:
#             e_gs = e0
#             gs = eigvecs[:, i0]
#             gs_sec = [n_up, n_dn]
# print(f"Ground state (E={e_gs:.2f}, sector {gs_sec}):")
# print(gs)

