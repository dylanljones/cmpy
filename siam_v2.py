#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import NamedTuple
from transonic import Array, boost, Dict, List, NDim, Tuple, Type
import numpy as np
from collections import defaultdict
from itertools import product, chain
from bisect import bisect_left
from scipy.sparse.linalg import LinearOperator




BINARY_REPR = True
IMPURITY = 0b1  # impurity site site is the first bit

NUM_TYP = Type(float, complex)
VECTOR = Array[NUM_TYP, NDim(1, 2)]  # need 2 dimensional for matrix@matrix
VECTOR1D = Array[NUM_TYP, NDim(1)]
VECTOR2D = Array[NUM_TYP, NDim(2)]
CVECTOR = Array[complex, NDim(1, 2)]
ND = NDim(1, 2)

PRM_ARRAY = Array[float, "1d"]
STATE_ARRAY = Array[int, "1d"]


class SpinState(NamedTuple):
    """Wrapper for integer/binary representation of states."""

    num: int

    def occ(self, pos: int) -> int:
        """Return particle number at `pos`."""
        return self.num & 1 << pos

    def occ_total(self) -> int:
        """Return total number of particles."""
        return bin(self.num).count("1")

    def to_array(self) -> Array[np.int8, "1d"]:
        """Convert binary representation to integer array."""
        return np.fromiter(f"{self.num:b}", dtype=np.int8)[::-1]

    def bitrep(self, digits=None) -> str:
        """Bit representation of the State."""
        return (f"{self.__class__.__name__}(num=0b"
                + (f"{self.num:0{digits}b}" if digits else f"{self.num:b}")
                + ")")

    def __repr__(self) -> str:
        """Represent state in binary if `BINARY_REPR` else as number."""
        if BINARY_REPR:
            return self.bitrep()
        return f"{self.__class__.__name__}(num={self.num})"


def create_spinstates(
        size: int, return_state=False
) -> Dict[int, STATE_ARRAY or List[SpinState]]:
    """Create `SpinState`s for a lattice with `size` sites.

    Parameters
    ----------
    size : int
        Number of sites
    return_state : bool, optional
        Weather to return a list of `SpinState` objects,
        or an integer `np.ndarray`.

    Returns
    -------
    spinstates : dict
        Dictionary containing an `int` array of all spinstates for any particle
        number.  The particle number are the `dict` keys.
        If `return_state`, the `dict` values are `list` of `SpinState` instead
        of an array.

    """
    spinstates = defaultdict(list)
    max_int = int('1' * size, base=2)

    for ii in range(max_int + 1):
        state = SpinState(ii)
        spinstates[state.occ_total()].append(state if return_state else state.num)
    spinstates = dict(spinstates)
    if return_state:
        return spinstates
    for key, val in spinstates.items():
        spinstates[key] = np.array(val)
    return spinstates


def calc_onsite_energy(e_onsite: PRM_ARRAY, num_spin: int) -> float:
    """Calculate the on-site energy for a `num_spin`.

    Parameters
    ----------
    e_onsite : (M) float np.ndarray
        On-site energies of the sites, `e_onsite[0]` corresponds to the impurity.
    num_spin : int
        Integer representation of the state.

    Returns
    -------
    onsite_energy : float
        Total on-site energy for a spin state.

    """
    state = np.array([int(ss) for ss in bin(num_spin)[:1:-1]], dtype=bool)
    return np.sum(e_onsite[:state.size] * state)


def calc_center_interaction(U: float, num_up: int, num_dn: int) -> float:
    """Calculate the local Hubbard interaction for site 0 only.

    Parameters
    ----------
    U : float
        Local interaction on the impurity site.
    num_up, num_dn : int
        Integer representation of the up and dn spin state

    Returns
    -------
    interaction_energy : float
        Interaction energy of the impurity site.

    """
    overlap = IMPURITY & num_up & num_dn
    return overlap * U


def _calc_hopp_from_center(V: PRM_ARRAY, num_spin: int) -> List[Tuple[float, int]]:
    """How you would implement it in python instead.
    e.g. [0 1 0 1]
         [0 1 0 0]   num_new

    """
    if not num_spin & IMPURITY:  # no electron that can hop
        return
    num_new = num_spin ^ IMPURITY  # remove center electron

    sign = +1  # every time we hop past an electron we pick up a - sign
    for nn, vn in enumerate(V, start=1):
        if num_spin & (1 << nn):  # state filled, no hopping but sign change
            sign *= -1
        else:
            yield sign*vn, num_new ^ (1 << nn)


def _calc_hopp_to_center(V: PRM_ARRAY, num_spin: int) -> List[Tuple[float, int]]:
    """How you would implement it in python instead.
    e.g.[0 1 1 0]
        [0 1 1 1]    num_new
    """
    if num_spin & IMPURITY:  # impurity already occupied
        return
    num_new = num_spin ^ IMPURITY  # add center electron

    sign = +1  # every time we hop past an electron we pick up a - sign
    for nn, vn in enumerate(V, start=1):
        if num_spin & (1 << nn):  # state filled, hopping possible
            yield sign * vn, num_new ^ (1 << nn)
        else:
            sign *= -1


def apply_up_onsite(matvec: VECTOR, x: VECTOR, e_onsite: PRM_ARRAY,
                    up_states: STATE_ARRAY, num_dn_states: int):
    """Apply the up-spin on-site term on state `x`, output is added to `matvec`.

    Parameters
    ----------
    matvec : (N) float or complex np.ndarray
        Output vector, result is added.
    x : (N) float or complex np.ndarray
        State to which the on-site term is applied.
    e_onsite : (M) float np.ndarray
        Array of on-site energy parameters, `e_onsite[0]` corresponds to the
        impurity site.
    up_states : "int[]"
        Array of integer representation of up states.
    num_dn_states : int
        Number of down-spin states.

    """
    # acts only on up spin -> can be applied for all dn spins with this up state
    dn_states = np.arange(num_dn_states)
    for num_up, state_up in enumerate(up_states):
        eps_up = calc_onsite_energy(e_onsite, num_spin=state_up)
        full_indices = num_up*num_dn_states + dn_states
        matvec[full_indices] += eps_up*x[full_indices]


def apply_dn_onsite(matvec: VECTOR, x: VECTOR, e_onsite: PRM_ARRAY,
                    num_up_states: int, dn_states: STATE_ARRAY):
    """Apply the dn-spin on-site term on state `x`, output is added to `matvec`.

    Parameters
    ----------
    matvec : (N) float or complex np.ndarray
        Output vector, result is added.
    x : (N) float or complex np.ndarray
        State to which the on-site term is applied.
    e_onsite : (M) float np.ndarray
        Array of on-site energy parameters, `e_onsite[0]` corresponds to the
        impurity site.
    num_up_states : int
        Number of up-spin states.
    dn_states : "int[]"
        Array of integer representation of down states.

    """
    # acts only on dn spin -> can be applied for all up spins with this dn state
    up_states = np.arange(num_up_states)
    for num_dn, state_dn in enumerate(dn_states):
        eps_dn = calc_onsite_energy(e_onsite, num_spin=state_dn)
        full_indices = up_states*len(dn_states) + num_dn
        matvec[full_indices] += eps_dn*x[full_indices]


def apply_center_interaction(matvec: VECTOR, x: VECTOR, U: float,
                             up_states: STATE_ARRAY, dn_states: STATE_ARRAY):
    """Apply the local interaction term of the impurity on state `x`, output is added to `matvec`.

    Parameters
    ----------
    matvec : (N) float or complex np.ndarray
        Output vector, result is added.
    x : (N) float or complex np.ndarray
        State to which the interaction term is applied.
    U : float
        Parameter for the local on-site integer of the impurity site.
    up_states, dn_states : "int[]"
        Array of integer representation of up and down states.

    """
    for element, (state_up, state_dn) in enumerate(product(up_states, dn_states)):
        interaction = calc_center_interaction(U, num_up=state_up, num_dn=state_dn)
        matvec[element] += interaction*x[element]


def apply_up_hopping(matvec: VECTOR, x: VECTOR, hybrid: PRM_ARRAY,
                     up_states: STATE_ARRAY, num_dn_states: int):
    """Apply the up-spin hopping term on state `x`, output is added to `matvec`.

    Parameters
    ----------
    matvec : (N) float or complex np.ndarray
        Output vector, result is added.
    x : (N) float or complex np.ndarray
        State to which the on-site term is applied.
    hybrid : (M-1) float np.ndarray
        Parameters for the hybridization (hopping). The parameter gives the
        hopping amplitude between a bath site and the impurity.
    up_states : "int[]"
        Array of integer representation of up states.
    num_dn_states : int
        Number of down-spin states.

    """
    dn_states = np.arange(num_dn_states)
    for num_up, state_up in enumerate(up_states):
        up_hopping = chain(_calc_hopp_to_center(hybrid, num_spin=state_up),
                     _calc_hopp_from_center(hybrid, num_spin=state_up))
        origin = num_up*num_dn_states + dn_states
        for amplitude, new_state in up_hopping:
            up_idx = bisect_left(up_states, new_state)
            new_full_indices = up_idx * num_dn_states + dn_states
            matvec[new_full_indices] += amplitude * x[origin]


def apply_dn_hopping(matvec: VECTOR, x: VECTOR, hybrid: PRM_ARRAY,
                     num_up_states: "int", dn_states: STATE_ARRAY):
    """Apply the dn-spin hopping term on state `x`, output is added to `matvec`.

    Parameters
    ----------
    matvec : (N) float or complex np.ndarray
        Output vector, result is added.
    x : (N) float or complex np.ndarray
        State to which the on-site term is applied.
    hybrid : (M-1) float np.ndarray
        Parameters for the hybridization (hopping). The parameter gives the
        hopping amplitude between a bath site and the impurity.
    num_up_states : int
        Number of up-spin states.
    dn_states : "int[]"
        Array of integer representation of down states.

    """
    up_states = np.arange(num_up_states)
    for num_dn, state_dn in enumerate(dn_states):
        dn_hoping = chain(_calc_hopp_to_center(hybrid, num_spin=state_dn),
                     _calc_hopp_from_center(hybrid, num_spin=state_dn))
        origin = up_states * len(dn_states) + num_dn
        for amplitude, new_state in dn_hoping:
            dn_idx = bisect_left(dn_states, new_state)
            new_full_indices = up_states*len(dn_states) + dn_idx
            matvec[new_full_indices] += amplitude * x[origin]


def siam_matvec(x: VECTOR, e_onsite: PRM_ARRAY, U: float, hybrid: PRM_ARRAY,
                up_states: STATE_ARRAY, dn_states: STATE_ARRAY) -> VECTOR:
    """Apply the SIAM Hamiltonian on state `x`.

    Parameters
    ----------
    x : (N) float or complex np.ndarray
        State to which the Hamiltonian is applied.
    e_onsite : (M) float np.ndarray
        Array of on-site energy parameters, `e_onsite[0]` corresponds to the
        impurity site.
    U : float
        Parameter for the local on-site integer of the impurity site.
    hybrid : (M-1) float np.ndarray
        Parameters for the hybridization (hopping). The parameter gives the
        hopping amplitude between a bath site and the impurity.
    up_states, dn_states : "int[]"
        Array of integer representation of up and down states.

    """
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


class SiamHamilton(LinearOperator):
    """Hamiltonian for the SIAM model / Hubbard Star in a fixed particle sector.

    Attributes
    ----------
    e_onsite : (M) float np.ndarray
        Array of on-site energy parameters, `e_onsite[0]` corresponds to the
        impurity site.
    U : float
        Parameter for the local on-site integer of the impurity site.
    hybrid : (M-1) float np.ndarray
        Parameters for the hybridization (hopping). The parameter gives the
        hopping amplitude between a bath site and the impurity.
    up_states, dn_states : "int[]"
        Array of integer representation of up and down states.

    """

    def __init__(self, e_onsite: PRM_ARRAY, U: float, hybrid: PRM_ARRAY,
                 up_states: STATE_ARRAY, dn_states: STATE_ARRAY):
        """Linear operator for the SIAM in a fixed particle sector.

        Parameters
        ----------
        e_onsite : (M) float np.ndarray
            Array of on-site energy parameters, `e_onsite[0]` corresponds to the
            impurity site.
        U : float
            Parameter for the local on-site integer of the impurity site.
        hybrid : (M-1) float np.ndarray
            Parameters for the hybridization (hopping). The parameter gives the
            hopping amplitude between a bath site and the impurity.
        up_states, dn_states : "int[]"
            Array of integer representation of up and down states.

        """
        # FIXME allow magnetic e_onsite and hybridization
        assert e_onsite.shape[-1] == hybrid.shape[-1] + 1
        assert np.asarray(U).size in (1, e_onsite.shape[-1])
        dim = len(up_states) * len(dn_states)
        super().__init__(dtype=complex, shape=(dim, dim))
        # make sure we have correct type
        self.e_onsite = np.asarray(e_onsite).astype(dtype=float, casting='safe')
        self.U = float(U)
        self.hybrid = np.asarray(hybrid).astype(dtype=float, casting='safe')
        # up/dn states as created by create_spinstates
        self.up_states = up_states
        self.dn_states = dn_states

    def _matvec(self, x: VECTOR) -> VECTOR:
        # FIXME: replace copy by ascountinousarray or something
        return siam_matvec(x.copy(), self.e_onsite, self.U, self.hybrid,
                           self.up_states, self.dn_states)

from numpy.random import default_rng
RNG = default_rng(0)

BETA = 17.3
N_SITES = 3
U = 2.
eps = 2*RNG.random(N_SITES) - 1
V = RNG.random(N_SITES - 1)
print('ϵ:', eps)
print('V:', V)

n_up = 1
n_dn = 1


states = create_spinstates(N_SITES)
ham = SiamHamilton(eps, U, hybrid=V,
                      up_states=states[n_up], dn_states=states[n_dn])
# small system, no Lanczos method necessary use full digitalization
ham_mat = ham @ np.eye(*ham.shape)
print(ham_mat.round(decimals=1))
# assert np.allclose(ham_mat, ham_mat.T.conj()), "Hamiltonian must be Hermitian."
eig, vec = np.linalg.eigh(ham_mat)




