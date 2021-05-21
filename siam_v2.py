#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import NamedTuple
from transonic import Array, boost, Dict, List, NDim, Tuple, Type
import numpy as np
from collections import defaultdict
from itertools import product, chain
from bisect import bisect_left
from scipy.sparse.linalg import LinearOperator
import gftool as gt




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
    """
    e.g. [0 1 0 1]
         [0 1 0 0]   num_new
    e.g.   c1+ c0 c0+ c1+ |0>  = c1+ c1+ |0> = 0
    e.g.   c2+ c0 c0+ c1+ |0>  = c2+ c1+ |0>  =  - c1+ c2+ |0>
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
    """
    e.g.[0 1 1 0]               c1+ c2+ |0>
        [0 1 1 1]    num_new    c0+ c1+ c2+ |0>
    e.g.   c0+ c1 c1+ c2+ |0>  =  c0+ c2+ |0>
    e.g.   c0+ c2 c1+ c2+ |0>  =  - c0+ c1+ c2 c2+ |0>  = - c0+ c1+ |0>
    """
    if num_spin & IMPURITY:  # impurity already occupied
        return
    num_new = num_spin ^ IMPURITY  # add center electron

    sign = +1  # every time we hop past an electron we pick up a - sign
    for nn, vn in enumerate(V, start=1):
        if num_spin & (1 << nn):  # state filled, hopping possible
            yield sign * vn, num_new ^ (1 << nn)
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
        dn_hopping = chain(_calc_hopp_to_center(hybrid, num_spin=state_dn),
                     _calc_hopp_from_center(hybrid, num_spin=state_dn))
        origin = up_states * len(dn_states) + num_dn
        for amplitude, new_state in dn_hopping:
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


def creation_0up_matvec(matvec: VECTOR, x: VECTOR, up_states: STATE_ARRAY,
                        up_p1_states: STATE_ARRAY, num_dn_states: int):
    """Apply the creation operator :math:`c^†_{0↑}`  on state `x`.

    Parameters
    ----------
    matvec : (N) float or complex np.ndarray
        Output containing the matrix vector product.
    x : (N) float or complex np.ndarray
        State to which the interaction term is applied.
    up_states, up_p1_states : (M) int np.ndarray
        Array of integer representation of up-states for the particle sector,
        and the particle sector with one electron more.
    num_dn_states : int
        Size of the dn-states particle sector.

    """
    dn_states = np.arange(num_dn_states)
    for num_up, state_up in enumerate(up_states):
        if state_up & IMPURITY:  # already particle, no creation possible
            continue
        new_up = state_up ^ IMPURITY  # add electron
        new_num = bisect_left(up_p1_states, new_up)
        origins = num_up*num_dn_states + dn_states
        matvec[new_num*num_dn_states + dn_states] += x[origins]


class MeasureUp:
    """Class facilitating the measurement of the up-spin impurity Green's function."""

    def __init__(self, beta: float, z: CVECTOR, states: Dict[int, STATE_ARRAY]):
        """Measurement object for the up-spin impurity Green's function.

        Parameters
        ----------
        beta : float
            Inverse temperature.
        z : CVECTOR
            Complex frequencies for which the Lehmann sum is evaluated.
        states : Dict[int, STATE_ARRAY]
            Dictionary of the states as created by `create_spinstates`.

        """
        self.beta = beta
        self.z = z
        self.states = states
        self.partition_ = 0
        self.gf_up_ = np.zeros_like(z)
        self.occ_0up_ = 0
        self.occ_0dbl_ = 0
        self.gs_energy: float = np.infty

    @property
    def partition_fct(self):
        r"""Partition function :math:`Tr\exp(-βH)`."""
        return self.partition_ * np.exp(-self.beta*self.gs_energy)

    @property
    def gf_up_z(self):
        r"""Up-spin impurity Green's function :math:`⟨⟨c_{0↑}|c†_{0↓}⟩⟩(z)`."""
        return self.gf_up_ / self.partition_

    @property
    def occ_0up(self):
        """Occupation of the up-spin impurity site :math:`⟨n_{0↑}⟩`."""
        return self.occ_0up_ / self.partition_

    @property
    def occ_0dbl(self):
        """Double occupation of the impurity site :math:`⟨n_{0↑}n_{0↓}⟩`."""
        return self.occ_0dbl_ / self.partition_

    def accumulate(self, eig, vec, eig_up_p1, vec_up_p1, n_up, n_dn):
        """Perform measurements to accumulate observables."""
        states = self.states
        min_energy = min(eig)
        if min_energy < self.gs_energy:
            factor = np.exp(-self.beta*(self.gs_energy - min_energy))
            self.gs_energy = min_energy
        else:
            factor = 1  # do nothing

        self.acc_partition(eig=eig, factor=factor)
        ocd_0up = CreationCenterUp(states[n_up], states[n_up+1], num_dn_states=len(states[n_dn]))
        ocd_0up_vec = ocd_0up.matmat(vec)
        self.acc_gf_up(eig, ocd_0up_vec, eig_p1up=eig_up_p1, vec_p1up=vec_up_p1, factor=factor)
        self.acc_occ_0up(eig, vec, up_states=states[n_up],
                         num_dn_states=len(states[n_dn]), factor=factor)
        self.acc_occ_0dbl(eig, vec, up_states=states[n_up],
                          dn_states=states[n_dn], factor=factor)

    def acc_partition(self, eig, factor=1):
        """Perform measurement to accumulate the partition function."""
        self.partition_ *= factor
        self.partition_ += np.sum(np.exp(-self.beta*(eig-self.gs_energy)))

    def acc_occ_0up(self, eig, vec, up_states: STATE_ARRAY, num_dn_states: int, factor=1):
        """Perform measurement to accumulate the up-spin impurity occupation."""
        dn_states = np.arange(num_dn_states)
        self.occ_0up_ *= factor
        for num_up, state_up in enumerate(up_states):
            if not state_up & IMPURITY:  # not occupied, nothing to count
                continue
            indices = num_up*num_dn_states + dn_states
            overlap = np.sum(abs(vec[indices, :])**2, axis=0)
            self.occ_0up_ += np.sum(np.exp(-self.beta*(eig - self.gs_energy)) * overlap)

    def acc_occ_0dbl(self, eig, vec, up_states: STATE_ARRAY, dn_states: STATE_ARRAY, factor=1):
        """Perform measurement to accumulate the impurity double occupation."""
        self.occ_0dbl_ *= factor
        for element, (state_up, state_dn) in enumerate(product(up_states, dn_states)):
            if not IMPURITY & state_up & state_dn:
                continue
            overlap = abs(vec[element, :])**2
            self.occ_0dbl_ += np.sum(np.exp(-self.beta*(eig - self.gs_energy)) * overlap)

    def acc_gf_up(self, eig, ocd_0up_vec, eig_p1up, vec_p1up, factor=1):
        """Accumulate up-spin 1-particle Green's function ⟨⟨c_{0↑}|c†_{0↓}⟩⟩(z)."""
        self.gf_up_ *= factor
        gf_up0_accumulate(self.gf_up_, self.z, self.beta,
                          eig, ocd_0up_vec, eig_p1up, vec_p1up, self.gs_energy)


class CreationCenterUp(LinearOperator):
    """Creation operator :math:`c^†_{0↑}` at site 0 for up electrons."""

    def __init__(self, up_states: STATE_ARRAY, up_p1_states: STATE_ARRAY, num_dn_states: int):
        """Linear operator operator for the creation operator :math:`c^†_{0↑}`.

        Parameters
        ----------
        up_states, up_p1_states : (M) int np.ndarray
            Array of integer representation of up-states for the particle sector,
            and the particle sector with one electron more.
        num_dn_states : int
            Size of the dn-states particle sector.

        """
        self.up_states = up_states
        self.up_p1_states = up_p1_states
        self.num_dn_states = num_dn_states
        dim_origin = len(up_states) * num_dn_states
        dim_target = len(up_p1_states) * num_dn_states
        super().__init__(dtype=np.complex, shape=(dim_target, dim_origin))

    def _matvec(self, x):
        newsize = len(self.up_p1_states)*self.num_dn_states
        matvec = np.zeros((newsize, *x.shape[1:]), dtype=x.dtype)
        # line handles vector shape. Whether matvec is of shape [N,] or [N,1]
        # v = np.zeros((4,)); v2 = np.zeros((4,1))
        # [*v.shape[1:]]  --> []
        # [*v2.shape[1:]]  --> [1]
        creation_0up_matvec(matvec, x.copy(), self.up_states, self.up_p1_states, self.num_dn_states)
        return matvec


def gf_up0_accumulate(values: CVECTOR, z: CVECTOR, beta: float,
                      eig: VECTOR1D, ocd_0up_vec: VECTOR2D,
                      eig_p1up: VECTOR1D, vec_p1up: VECTOR2D,
                      gs_energy: float):
    """Accumulate up-spin impurity Green's function using Lehmann sum.

    Parameters
    ----------
    values : CVECTOR
        Output array for the accumulated Green's function.
    z : CVECTOR
        z
    z : CVECTOR
        Complex frequencies for which the Lehmann sum is evaluated.
    beta : float
        Inverse temperature.
    eig, eig_p1up : VECTOR1D
        Eigenvalues for and eigenvalues for an additional up-spin electron.
    ocd_0up_vec, vec_p1up : VECTOR2D
        Creation operator applied to eigenvectors, and eigenvectors for an
        additional up-spin electron.
    gs_energy : float
        Ground state energy, that is the smallest eigenvalue.

    """
    overlap = vec_p1up.T.conj() @ ocd_0up_vec
    overlap = vec_p1up.conj() @ ocd_0up_vec.T
    overlap = abs(overlap)**2
    exp_en_vec = np.exp(-beta*(eig - gs_energy))
    for mm, eig_m in enumerate(eig_p1up):
        exp_em = np.exp(-beta*(eig_m - gs_energy))
        for nn, eig_n in enumerate(eig):
            denom = (z + eig_n - eig_m)
            values += overlap[mm, nn] * (exp_en_vec[nn] + exp_em) / denom



from numpy.random import default_rng
RNG = default_rng(0)

BETA = 17.3
N_SITES = 3
U = 2.
eps = np.ones(N_SITES) - 0.5  # 2 * RNG.random(N_SITES) - 1
V = -2 * np.ones(N_SITES - 1)  # RNG.random(N_SITES - 1)
print('ϵ:', eps)
print('V:', V)

n_up = 1
n_dn = 1

# Pade frequencies
izp, rp = gt.pade_frequencies(200, beta=BETA)
# Matsubara frequencies
iw = gt.matsubara_frequencies(range(200), beta=BETA)
# Real axis
ww = np.linspace(-2, 2, num=200) + 1e-2j



states = create_spinstates(N_SITES)
ham = SiamHamilton(eps, U, hybrid=V,
                      up_states=states[n_up], dn_states=states[n_dn])
# small system, no Lanczos method necessary use full digitalization
ham_mat = ham @ np.eye(*ham.shape)
print(ham_mat.round(decimals=1))
assert np.allclose(ham_mat, ham_mat.T.conj()), "Hamiltonian must be Hermitian."
eig, vec = np.linalg.eigh(ham_mat)


partition = 0  # accumulate Z = ∑_n ⟨n|exp(-βH)|n⟩ directly
ln_partition = -np.infty  # accumulate logarithm of partition
measure_up = MeasureUp(beta=BETA, z=ww, states=states)


states = create_spinstates(3, return_state=True)
for i in states:
    print(states[i])


