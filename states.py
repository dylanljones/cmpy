#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cmpy.basis as basis


def _dec_to_bin(state, site_num):
    return format(state, f'0{site_num}b')


def dec_to_bin(state):
    return bin(state)[2:]


def bin_to_dec(binary):
    return int(binary, 2)


def electron_count(state):
    return bin(state).count("1")


def _occupied(state, site):
    occ = False
    state_bin = dec_to_bin(state)
    occ = bool(state_bin[-site - 1])
    return occ


def occupied(state, site):
    return bool(state & (1 << site))


def get_all_states(site_num, particle_num):
    states = []
    for i in range(0, 2 ** (site_num)):
        # print(dec_to_bin(i))
        if electron_count(i) == particle_num:
            states.append(i)
    return states


def _state_to_vector(state, site_num):
    state_bin = dec_to_bin(state, site_num)
    state_vec = [int(d) for d in state_bin]
    return np.array(state_vec)


def state_to_vector(state):
    state_vec = [int(d) for d in dec_to_bin(state)]
    return np.array(state_vec, dtype=np.bool_)


def vector_to_state(vector):
    state_bin = vector_to_bin(vector)
    state_vec = bin_to_dec(state_bin)
    return state_vec


def vector_to_bin(state_vec):
    return ''.join(str(e) for e in state_vec)


def get_fullstate_num(up_state_num, dn_state_num, site_num):
    return up_state_num * site_num + dn_state_num


def get_state_num(fullstate_num):
    up_state_num = fullstate_num / 2
    dn_state_num = fullstate_num % 2
    return up_state_num, dn_state_num


def print_states(states, site_num):
    state_bins = []
    for state in states:
        state_bins.append(dec_to_bin(state, site_num))
    print("state_num", "state_bin", "state")
    for i, j in enumerate(state_bins):
        print(i, "\t\t", j, "\t\t", bin_to_dec(j))


def print_fullstates(up_states, dn_states, site_num):
    print("up_idx", "|", "up", "|", "up_bin", "|", "dn_idx", "|", "dn", "|", "dn_bin", "|", "fullstate_num")
    for up_idx, up in enumerate(up_states):
        for dn_idx, dn in enumerate(dn_states):
            up_bin = dec_to_bin(up, site_num)
            dn_bin = dec_to_bin(dn, site_num)
            full = get_fullstate_num(up_idx, dn_idx, site_num)
            print(up_idx, "\t\t", up, "\t\t", up_bin, "\t\t", dn_idx, "\t\t", dn, "\t\t", dn_bin, "\t\t", full)


def _compute_onsite(eps_onsite, state, site_num):
    state_vec = state_to_vector(state, site_num)
    return np.dot(eps_onsite, state_vec)


def compute_onsite(state, eps_onsite):
    rev_bit_array = state_to_vector(state)[::-1]
    return np.sum(eps_onsite[:rev_bit_array.size] * rev_bit_array)


IMPURITY = 0b1
site_num = 4
eps_onsite = [1, 2, 3, 4]
U = 4
V = [1, 2, 3, 4]


def compute_interaction(up_state, dn_state, U):
    overlap = IMPURITY & up_state & dn_state
    return overlap * U


def compute_hopping(V, num_spin, target_site):
    if not num_spin & IMPURITY or num_spin & (1 << target_site):  # no electron that can hop
        return
    num_new = num_spin ^ IMPURITY  # remove center electron
    sign = +1
    for kk in range(1, target_site + 1):
        if num_spin & (1 << kk):
            sign *= -1
    return sign * V[kk], num_new ^ (1 << kk)


def get_hopping_el(up, dn, full, up_idx, dn_idx):
    hopp_up_elements = []
    hopp_dn_elements = []
    for site in range(1, site_num):
        hopp_up_e, new_up_state = compute_hopping(V, up, site)
        full_hopp_up = get_fullstate_num(new_up_state, dn_idx)
        hopp_up_elements.append([full_hopp_up, full, hopp_up_e])

        hopp_dn_e, new_dn_state = compute_hopping(V, dn, site)
        full_hopp_dn = get_fullstate_num(up_idx, new_dn_state)
        hopp_dn_elements.append([full_hopp_dn, full, hopp_dn_e])

    return hopp_dn_elements + hopp_up_elements


def get_onsite_elements(up_states, dn_states, site_num, eps_onsite):
    onsite_elements = []
    for up_idx, up in enumerate(up_states):
        for dn_idx, dn in enumerate(dn_states):
            full = get_fullstate_num(up_idx, dn_idx, site_num)
            e_up = compute_onsite(eps_onsite, up)
            e_dn = compute_onsite(eps_onsite, dn)
            onsite_elements.append([full, full, e_up + e_dn])
    return np.array(onsite_elements)


def get_onsite_el(up, dn, full):
    e_up = compute_onsite(eps_onsite, up)
    e_dn = compute_onsite(eps_onsite, dn)
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
            full = get_fullstate_num(up_idx, dn_idx, site_num)

            onsite_elements.append(get_onsite_el(up, dn, full))

            interaction_elements.append(get_inter_el(up, dn, full))

            hopping_elements += get_hopping_el(up, dn, full, up_idx, dn_idx)
    return onsite_elements + interaction_elements + hopping_elements


def _onsite(up_states, dn_states, site_num, eps_onsite):
    onsite_elements = []
    for up_idx, up in enumerate(up_states):
        for dn_idx, dn in enumerate(dn_states):
            full = get_fullstate_num(up_idx, dn_idx, site_num)
            e_up = _compute_onsite(eps_onsite, up, site_num)
            e_dn = _compute_onsite(eps_onsite, dn, site_num)
            onsite_elements.append([full, full, e_up + e_dn])
    return np.array(onsite_elements)


def interaction(up_states, dn_states, site_num, u):
    interaction_elements = []
    for up_idx, up in enumerate(up_states):
        for dn_idx, dn in enumerate(dn_states):
            full = get_fullstate_num(up_idx, dn_idx, site_num)
            up_vec = state_to_vector(up, site_num)
            dn_vec = state_to_vector(dn, site_num)
            interaction_e = (up_vec * dn_vec)[0] * u
            interaction_elements.append([full, full, interaction_e])
    return np.array(interaction_elements)


def hopping(up_states, dn_states, site_num, v):
    # v_up = [5] * (site_num - 1)
    hopping_elements = []
    for k in range(1, site_num):
        for up_idx, up in enumerate(up_states):
            for dn_idx, dn in enumerate(dn_states):
                full = get_fullstate_num(up_idx, dn_idx, site_num)
                up_bin = dec_to_bin(up, site_num)
                dn_bin = dec_to_bin(dn, site_num)
                up_vec = state_to_vector(up, site_num)
                dn_vec = state_to_vector(dn, site_num)
                # print(up_bin[0], up_bin[1])

                if int(up_bin[0]) and not int(up_bin[k]):
                    new_up_vec = up_vec
                    new_up_vec[0] = 0
                    new_up_vec[k] = 1
                    new_up_state = vector_to_state(new_up_vec)
                    new_up_idx = up_states.index(new_up_state)

                    # print(up_bin, "->", new_up_vec, new_up_idx)
                    new_up_full = get_fullstate_num(new_up_idx, dn_idx, site_num)
                    hopping_elements.append([new_up_full, full, v])

                if not int(up_bin[0]) and int(up_bin[k]):
                    new_up_vec = up_vec
                    new_up_vec[0] = 1
                    new_up_vec[k] = 0
                    new_up_state = vector_to_state(new_up_vec)
                    new_up_idx = up_states.index(new_up_state)

                    new_up_full = get_fullstate_num(new_up_idx, dn_idx, site_num)
                    hopping_elements.append([new_up_full, full, v])

                if int(dn_bin[0]) and not int(dn_bin[k]):
                    new_dn_vec = dn_vec
                    new_dn_vec[0] = 0
                    new_dn_vec[k] = 1
                    new_dn_state = vector_to_state(new_dn_vec)
                    new_dn_idx = dn_states.index(new_dn_state)

                    new_dn_full = get_fullstate_num(up_idx, new_dn_idx, site_num)
                    hopping_elements.append([new_dn_full, full, v])

                if not int(dn_bin[0]) and int(dn_bin[k]):
                    new_dn_vec = dn_vec
                    new_dn_vec[0] = 1
                    new_dn_vec[k] = 0
                    new_dn_state = vector_to_state(new_dn_vec)
                    new_dn_idx = dn_states.index(new_dn_state)

                    new_dn_full = get_fullstate_num(up_idx, new_dn_idx, site_num)
                    hopping_elements.append([new_dn_full, full, v])

    return np.array(hopping_elements)


def set_hamiltonian(up_states, dn_states, site_num, eps_onsite, u, v):
    onsite_el = _onsite(up_states, dn_states, site_num, eps_onsite)
    inter_el = interaction(up_states, dn_states, site_num, u)
    hopping_el = hopping(up_states, dn_states, site_num, v)
    all_elements = np.vstack((onsite_el, inter_el, hopping_el))
    col = all_elements[:, 1]
    row = all_elements[:, 0]
    data = all_elements[:, 2]
    dim = len(up_states) * len(dn_states)
    from scipy.sparse import csr_matrix
    hamiltonian = csr_matrix((data, (row, col)), shape=(dim, dim)).toarray()
    print(np.shape(hamiltonian))
    return hamiltonian


def get_groundstate(site_num):
    # site_num = 10
    num_up = 2
    num_dn = 2
    eps_b = 2
    eps_i = 3
    u = 99
    v = 5
    eps_onsite = [eps_i] + [eps_b] * (site_num - 1)

    gs = None
    e_gs = np.infty
    gs_sec = None

    for n_up in range(1, site_num):
        for n_dn in range(1, site_num):
            up_states = get_all_states(site_num, n_up)
            dn_states = get_all_states(site_num, n_dn)
            hamiltonian = set_hamiltonian(up_states, dn_states, site_num, eps_onsite, u, v)
            eigvals, eigvecs = np.linalg.eigh(hamiltonian)
            i0 = np.argmin(eigvals)
            e0 = eigvals[i0]
            if e0 < e_gs:
                e_gs = e0
                gs = eigvecs[:, i0]
                gs_sec = [n_up, n_dn]
    print(f"Ground state (E={e_gs:.2f}, sector {gs_sec}):")
    print(gs)


def test():
    site_num = 4
    num_up = 2
    num_dn = 2
    eps_b = 2
    eps_i = 3
    u = 99
    v = 5
    eps_onsite = [eps_i] + [eps_b] * (site_num - 1)
    up_states = get_all_states(site_num, num_up)
    dn_states = get_all_states(site_num, num_dn)
    print_states(up_states, site_num)
    print_states(dn_states, site_num)
    print_fullstates(up_states, dn_states, site_num)

    # print(onsite(up_states, dn_states, site_num, eps_onsite))
    # print(interaction(up_states, dn_states, site_num, u))
    # print(hopping(up_states, dn_states, site_num, v))
    set_hamiltonian(up_states, dn_states, site_num, eps_onsite, u, v)


def main():
    get_groundstate(4)


if __name__ == "__main__":
    # test()
    main()
