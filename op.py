#!/usr/bin/env python3
# -*- coding: utf-8 -*-

    from scipy.sparse.linalg import LinearOperator
    import numpy as np


def mv(v):
    return np.array([3 * v[0], 3 * v[1]])


A = LinearOperator((2, 2), matvec=mv)

print(A.matvec([3, 4]))

print(A * np.ones(2))


class SiamHamilton(LinearOperator):
    def __init__(self, e_onsite, U, hybrid, up_states, dn_states):
        dim = len(up_states) * len(dn_states)
        super().__init__(dtype=complex, shape=(dim, dim))
        # make sure we have correct type
        self.e_onsite = np.asarray(e_onsite).astype(dtype=float, casting='safe')
        self.U = float(U)
        self.hybrid = np.asarray(hybrid).astype(dtype=float, casting='safe')
        # up/dn states as created by create_spinstates
        self.up_states = up_states
        self.dn_states = dn_states

    def _matvec(self, x):
        # FIXME: replace copy by ascountinousarray or something
        return siam_matvec(x.copy(), self.e_onsite, self.U, self.hybrid,
                           self.up_states, self.dn_states)

def siam_matvec(x, e_onsite, U, hybrid, up_states, dn_states):
    # we label the states as n_up*len(n_dn) + n_up
    matvec = np.zeros_like(x)
    # diagonal terms:
    # eps_dn
    apply_dn_onsite(matvec, x, e_onsite, len(up_states), dn_states)
    # eps_up
    apply_up_onsite(matvec, x, e_onsite, up_states, len(dn_states))
    # interaction
    apply_center_interaction(matvec, x, U, up_states, dn_states)
    # dn-spin hopping
    apply_dn_hopping(matvec, x, hybrid, len(up_states), dn_states)
    # up-spin hopping
    apply_up_hopping(matvec, x, hybrid, up_states, len(dn_states))
    return matvec

ham = ed.SiamHamilton(eps, U, hybrid=V, up_states=states[n_up], dn_states=states[n_dn])
# small system, no Lanczos method necessary use full digitalization
ham_mat = ham @ np.eye(*ham.shape)
assert np.allclose(ham_mat, ham_mat.T.conj()), "Hamiltonian must be Hermitian."
eig, vec = np.linalg.eigh(ham_mat)