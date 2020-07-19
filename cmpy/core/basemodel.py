# coding: utf-8
"""
Created on 06 Jul 2020
Author: Dylan Jones

Abstract base classes for physical models.
"""
from abc import ABC, abstractmethod
from .basis import FockBasis
from .operators import HamiltonOperator
from typing import Any, ItemsView


class ModelParameters(object):

    """ Parameter class for storing parameters of physical models.

     The parameters can be accessed as attributes or dict-entries.
     This class is usually used as a base-class for model-classes.
     """

    def __init__(self, **params):
        """ Initializes the ModelParameters-instance.

        Parameters
        ----------
        **params: Initial parameters.
        """
        super(object, self).__init__()
        self.__params__ = dict(params)

    @property
    def params(self):
        """ dict: Returns a dictionary of all parameters. """
        return self.__params__

    def items(self) -> ItemsView:
        """ ItemsView: Returns the items of the parameter dictionary. """
        return self.__params__.items()

    def set_param(self, key: str, value: Any):
        """ Sets a parameter

        Parameters
        ----------
        key: str
            The name of the parameter.
        value: Any
            The value of the parameter.
        """
        self.__params__[key] = value

    def update_param(self, **params: dict):
        """ Update the parameters """
        self.__params__.update(params)

    def del_param(self, key: str):
        """ Deletes a parameter with the given name

        Parameters
        ----------
        key: str
            The name of the parameter to delete.
        """
        del self.__params__[key]

    def rename_param(self, key, new_key):
        """ Renames an existing parameter.

        Parameters
        ----------
        key: str
            The current name of the parameter.
        new_key: str
            The new name of the parameter.
        """
        self.__params__[new_key] = self.__params__[key]
        del self.__params__[key]

    def __getattribute__(self, key):
        """ Make parameters accessable as attributes. """
        key = str(key)
        if not key.startswith("__") and key in self.__params__.keys():
            return self.__params__[key]
        else:
            return super().__getattribute__(key)

    def __setattr__(self, key, value):
        """ Make parameters accessable as attributes. """
        key = str(key)
        if not key.startswith("__") and key in self.__params__.keys():
            self.__params__[key] = value
        else:
            super().__setattr__(key, value)

    def __getitem__(self, key):
        """ Make parameters accessable as dictionary items. """
        return self.__params__[key]

    def __setitem__(self, key, value):
        """ Make parameters accessable as dictionary items. """
        self.__params__[key] = value

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self.__params__)})"

    def __str__(self):
        return ", ".join([f"{k}={v}" for k, v in self.__params__.items()])


# noinspection PyAttributeOutsideInit
class AbstractModel(ModelParameters, ABC):

    """ Abstract base class for model classes.

    The AbstractModel-class derives from ModelParameters.
    All parameters are accessable as attributes or dictionary-items.
    """

    def __init__(self, **params):
        """ Initializes the AbstractModel-instance with the given initial parameters.

        Parameters
        ----------
        **params: Initial parameters of the model.
        """
        ModelParameters.__init__(self, **params)
        ABC.__init__(self)

    def __str__(self):
        return f"{self.__class__.__name__}({ModelParameters.__str__(self)})"


# noinspection PyAttributeOutsideInit
class AbstractBasisModel(AbstractModel):

    def __init__(self, num_sites=0, **params):
        super(AbstractModel, self).__init__(**params)
        self.basis = FockBasis(num_sites)

    @property
    def num_sites(self):
        return self.basis.num_sites

    @property
    def sector_keys(self):
        return self.basis.fillings

    @property
    def basis_labels(self):
        return self.basis.get_labels(bra=False, ket=False)

    def iter_sector_keys(self, repeat=2):
        return self.basis.iter_fillings(repeat)

    def iter_sectors(self):
        return self.basis.iter_sectors()

    def get_sector(self, n_up=None, n_dn=None, sector=None):
        return self.basis.get_sector(n_up, n_dn, sector)

    def build_matvec(self, matvec, x, sector):
        pass

    def hamiltonian_op(self, n_up=None, n_dn=None, sector=None):
        return HamiltonOperator(self, self.get_sector(n_up, n_dn, sector))

    def hamiltonian(self, n_up=None, n_dn=None, sector=None, x=None):
        hamop = HamiltonOperator(self, self.get_sector(n_up, n_dn, sector))
        return hamop.matrix(x)
