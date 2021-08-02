# coding: utf-8
#
# This code is part of cmpy.
# 
# Copyright (c) 2021, Dylan Jones

import logging
import logging.config

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
