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


def split_num(number: int, fractions: Union[float, Sequence[float]]) -> np.ndarray:
    """Splits an integer into integer fractions.

    Parameters
    ----------
    number : int
        The number to split. Must be larger then zero.
    fractions : float or Sequence of float
        The fraction sizes of the parts the number will be split into.

    Returns
    -------
    numbers : np.ndarray
        The integers the original number is split into. The sum is equal to the original number.
    """
    if isinstance(fractions, float):
        fractions = [fractions, 1-fractions]
    if sum(fractions) != 1:
        raise ValueError("Fractions have to add up to 1!")
    parts = (number * np.array(fractions)).astype(np.int64)
    parts[-1] += number - sum(parts)
    return parts


def create_subst_array(size: int, conc: Union[float, Sequence[float]],
                       values: Sequence[float]) -> np.ndarray:
    """Creates an (ordered) array of values.

    Parameters
    ----------
    size : int
        The size of the output array.
    conc : float or Sequence of float
        The concentrations of the values. If a single concentration is given
        it is interpreted as the concentration of the first of two values.
    values : Sequence of float
        The values for filling the array. The size must match the size of the concentrations.
        If one concentration is given the value-array must be of size 2.

    Returns
    -------
    array : np.ndarray
        The (ordered) array filled with the given values.
    """
    sizes = split_num(size, conc)
    arrays = [np.full(size, val) for size, val in zip(sizes, values)]
    return np.concatenate(arrays)


def random_permutations(arr, size, replace=False):
    """Creates (optionally unique) permutations of a given array.

    Parameters
    ----------
    arr : (N) np.ndarray
        The input array to permute.
    size : int
        The number of permutations to generate.
    replace : bool, optional
        If `True`, only unique permutations are returned. The default is `True`.

    Yields
    ------
    perm : (N) np.ndarray
        The permuted array.

    Examples
    --------
    >>> np.random.seed(0)
    >>> a = np.array([0, 0, 1, 1, 1])
    >>> perm = random_permutations(a, 2)
    >>> next(perm)
    array([1, 0, 0, 1, 1])
    >>> next(perm)
    array([0, 1, 1, 1, 0])
    """
    p = arr.copy()
    seen = set()
    count = 0
    while True:
        if count >= size:
            break
        np.random.shuffle(p)
        if not replace:
            phash = hash(str(p))
            if phash not in seen:
                seen.add(phash)
                yield p
                count += 1
        else:
            yield p
            count += 1


def subst_disorder_iter(size: int, conc: Union[float, Sequence[float]],
                        values: Sequence[float], samples: int):
    ordered = create_subst_array(size, conc, values)
    return random_permutations(ordered, samples)
