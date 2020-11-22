# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2020, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

from numpy.testing import assert_array_equal
from pytest import mark
from cmpy import utils


@mark.parametrize("items, cycle, result", [
    ([0, 1, 2], False, [[0, 1], [1, 2]]),
    ([0, 1, 2],  True, [[0, 1], [1, 2], [2, 0]]),
    (["0", "1", "2"], False, [["0", "1"], ["1", "2"]]),
    (["0", "1", "2"],  True, [["0", "1"], ["1", "2"], ["2", "0"]]),
])
def test_chain(items, cycle, result):
    assert_array_equal(utils.chain(items, cycle), result)
