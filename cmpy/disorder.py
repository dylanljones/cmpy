# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""This module contains methods for modeling disorder."""

import numpy as np
from typing import Union, Sequence


def create_subst_array(size: int, values: Sequence[float],
                       conc: Union[float, Sequence[float]]) -> np.ndarray:
    """Creates an (ordered) array of values.

    Parameters
    ----------
    size : int
        The size of the output array.
    values : Sequence of float
        The values for filling the array. The size must match the size of the concentrations.
        If one concentration is given the value-array must be of size 2.
    conc : float or Sequence of float
        The concentrations of the values. If a single concentration is given
        it is interpreted as the concentration of the first of two values.

    Returns
    -------
    array : np.ndarray
        The (ordered) array filled with the given values.
    """
    # Get sizes of sub-arrays
    if isinstance(conc, float):
        conc = [conc, 1-conc]
    if sum(conc) != 1:
        raise ValueError("Fractions have to add up to 1!")
    sizes = (size * np.array(conc)).astype(np.int64)
    sizes[-1] += size - sum(sizes)

    # create sub-arrays
    arrays = [np.full(size, val) for size, val in zip(sizes, values)]
    return np.concatenate(arrays)


def random_permutations(arr: Sequence[float], size: int, replace: bool = False, seed: int = None):
    """Creates (optionally unique) permutations of a given array.

    Parameters
    ----------
    arr : (N) np.ndarray
        The input array to permute.
    size : int
        The number of permutations to generate.
    replace : bool, optional
        If `True`, only unique permutations are returned. The default is `True`.
    seed : int, optional
        A optional seed to initialize the random number generator.

    Yields
    ------
    perm : (N) np.ndarray
        The permuted array.

    Examples
    --------
    >>> a = [0, 0, 1, 1, 1]
    >>> perm = random_permutations(a, size=2, seed=0)
    >>> next(perm)
    array([1, 1, 1, 0, 0])
    >>> next(perm)
    array([0, 1, 1, 1, 0])
    """
    rng = np.random.default_rng(seed)

    p = np.array(arr)
    seen = set()
    count = 0
    while True:
        if count >= size:
            break
        rng.shuffle(p)
        if not replace:
            phash = hash(p.data.tobytes())
            if phash not in seen:
                seen.add(phash)
                yield p
                count += 1
        else:
            yield p
            count += 1


def disorder_generator(size: int, values: Sequence[float], conc: Union[float, Sequence[float]],
                       samples: int, replace: bool = False, seed=None):
    """Generates (optionally unique) random samples from a given 1-D array.

    See Also
    --------
    random_permutations

    Parameters
    ----------
    size : int
        The size of the output array.
    values : Sequence of float
        The values for filling the array. The size must match the size of the concentrations.
        If one concentration is given the value-array must be of size 2.
    conc : float or Sequence of float
        The concentrations of the values. If a single concentration is given
        it is interpreted as the concentration of the first of two values.
    samples : int
        The number of random arrays to generate.
    replace : bool, optional
        If `True`, only unique permutations are returned. The default is `True`.
    seed : int, optional
        A optional seed to initialize the random number generator.

    Yields
    ------
    perm : (N) np.ndarray
        The randomly sampled arrays.

    Examples
    --------
    >>> eps = disorder_generator(5, values=[0, +1], conc=[0.4, 0.6], samples=2, seed=0)
    >>> next(eps)
    array([1, 1, 1, 0, 0])
    >>> next(eps)
    array([0, 1, 1, 1, 0])
    """
    ordered = create_subst_array(size, values, conc)
    return random_permutations(ordered, samples, replace, seed)
