# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2021, Dylan Jones

import logging
import collections
from abc import ABC, abstractmethod

# =========================================================================
# LOGGING
# =========================================================================

# Configure package logger
logger = logging.getLogger("cmpy")

_stream_handler = logging.StreamHandler()
_stream_handler.setLevel(logging.DEBUG)

# _frmt_str = "[%(asctime)s] %(levelname)-8s - %(name)s - %(funcName)s - %(message)s"
_frmt_str = "[%(asctime)s] %(name)s:%(levelname)-8s - %(message)s"
_formatter = logging.Formatter(_frmt_str, datefmt='%H:%M:%S')

_stream_handler.setFormatter(_formatter)    # Add formatter to stream handler
logger.addHandler(_stream_handler)          # Add stream handler to package logger

logger.setLevel(logging.WARNING)            # Set initial logging level
logging.root.setLevel(logging.NOTSET)


# =========================================================================
# ARRAY MIXIN's
# =========================================================================


class ArrayMixin(ABC):

    @abstractmethod
    def __getstate__(self):
        pass

    @abstractmethod
    def __setstate__(self, state):
        pass

    def copy(self):
        instance = self.__class__()
        instance.__setstate__(self.__getstate__())
        return instance

    def __copy__(self):
        return self.copy()

    def __format__(self, fstr):
        val_str = ", ".join([f"{x:{fstr}}" for x in self.__iter__()])
        return f"[{val_str}]"

    def __str__(self):
        return self.__format__("")


class Array(collections.abc.Sequence, ArrayMixin):
    pass


class MutableArray(collections.abc.MutableSequence, ArrayMixin):
    pass
