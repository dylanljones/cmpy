# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import numpy as np
from typing import Optional, Union, Any, Sequence, Dict
from cmpy.basis import UP
from cmpy.operators import project_onsite_energy, project_interaction, project_site_hopping
from cmpy.exactdiag import greens_function_lehmann
from .abc import AbstractManyBodyModel
import cmpy.models.tightbinding as tb
from cmpy import matshow

class HubbardModel(AbstractManyBodyModel):

    def __init__(self, u: Union[float, Sequence[Union[float]]] = 2.0,
                 eps: Union[float, Sequence[float]] = 0.0,
                 t: Union[float, Sequence[float]] = 1.0,
                 mu: Optional[float] = 0.0,
                 temp: Optional[float] = 0.0,
                 num_sites = 2):
        """Initializes the ``HubbardModel``.

        u: float or Sequence, optional
            The onsite interaction energy of the model. The default value is ``2``.
        eps: float or Sequence, optional
            The onsite energy of the model. The default value is ``0``.
        eps_bath: float or Sequence, optional
            The onsite energy of the model. The default value is ``0``.
        t: float or Sequence, optional
            The hopping parameter of the model. The default value is ``1``.
        mu: float, optional
            Optional chemical potential. The default is ``0``.
        temp: float, optional
            Optional temperature in kelvin. The default is ``0``.
        """
        super().__init__(u=u, eps=eps, t=t, mu=mu, temp=temp, num_sites=len(eps))

    def pformat(self):
        return f"U={self.u}, ε={self.eps}, t={self.t}, μ={self.mu}, T={self.temp}"

    def _hamiltonian_data(self, up_states, dn_states):
        """Gets called by the `hamilton_operator`-method of the abstract base class."""
        num_sites = len(self.eps)
        hoppings = create_hoppings(num_sites, self.t)

        yield from project_onsite_energy(up_states, dn_states, self.eps)
        yield from project_interaction(up_states, dn_states, self.u)
        for i in range(num_sites):
            yield from project_site_hopping(up_states, dn_states, num_sites, hoppings, pos=i)


def create_hoppings(num_sites, t):
    model = tb.BaseTightBindingModel(np.eye(1))
    model.add_atom()
    model.add_connections(1)
    model.build(num_sites, relative=True, periodic=0)
    model.set_hopping(t)
    ham_sparse = model.hamiltonian()
    ham = ham_sparse.toarray()
    return lambda i, j: ham[i, j]