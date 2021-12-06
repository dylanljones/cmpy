# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

from ._utils import logger, Plot, setup_plot

from .collection import *

from .basis import (
    UP, DN, SPIN_CHARS,
    state_label,
    binstr,
    binarr,
    binidx,
    overlap,
    occupations,
    create,
    annihilate,
    SpinState,
    State,
    Sector,
    Basis,
)

from .matrix import (
    matshow,
    transpose,
    hermitian,
    is_hermitian,
    diagonal,
    fill_diagonal,
    Decomposition,
    EigenState,
    Matrix
)

from .operators import (
    project_up,
    project_dn,
    project_elements_up,
    project_elements_dn,
    project_onsite_energy,
    project_interaction,
    project_site_hopping,
    project_hopping,
    LinearOperator,
    HamiltonOperator,
    TimeEvolutionOperator,
    CreationOperator,
    AnnihilationOperator
)

from .models.abc import (
    ModelParameters,
    AbstractModel,
    AbstractManyBodyModel
)

from . import _version
__version__ = _version.get_versions()['version']
