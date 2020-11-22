# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2020, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

import numpy as np
from lattpy import Lattice, Atom
from typing import Optional, Union, Any, Sequence, Dict
from .abc import AbstractModel


class HubbardModel(AbstractModel):

    def __init__(self, u: Union[float, Sequence[Union[float]]] = 2.0,
                 eps: Union[float, Sequence[float]] = 0.0,
                 t: Union[float, Sequence[float]] = 1.0,
                 mu: Optional[float] = 0.0,
                 temp: Optional[float] = 0.0):
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
        super().__init__(u=u, eps=eps, t=t, mu=mu, temp=temp)

    def pformat(self):
        return f"U={self.u}, ε={self.eps}, t={self.t}, μ={self.mu}, T={self.temp}"
