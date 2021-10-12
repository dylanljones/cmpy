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
    """Splits an integer into fractions.

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


def create_substitutional_array(size: int, conc: Union[float, Sequence[float]],
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
