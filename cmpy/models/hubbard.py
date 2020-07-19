# coding: utf-8
"""
Created on 07 Jul 2020
Author: Dylan Jones
"""
import numpy as np
from cmpy.core import AbstractModel, AbstractBasisModel


# noinspection PyAttributeOutsideInit
class HubbardModel(AbstractBasisModel):

    def __init__(self, u=1.0, t=1.0, eps=0., mu=None):
        mu = u / 2 if mu is None else mu
        super().__init__(u=u, eps=eps, t=t, mu=mu)

    def __str__(self):
        paramstr = f"U={self.u}, ε={self.eps}, t={self.t}, μ={self.mu}"
        return f"{self.__class__.__name__}({paramstr})"
