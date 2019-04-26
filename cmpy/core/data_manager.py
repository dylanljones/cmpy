# -*- coding: utf-8 -*-
"""
Created on 25 Apr 2019
@author: Dylan Jones

project: cmpy
version: 1.0

"""
import re
import os
from os.path import abspath, dirname, join

PROJECT_DIR = dirname(dirname(dirname(abspath(__file__))))
DATA_DIR = join(PROJECT_DIR, "_data")

def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

create_dir(DATA_DIR)


def list_files(root):
    results = list()
    for name in os.listdir(root):
        path = join(root, name)
        if os.path.isfile(path):
            results.append(path)
    return results

def list_dirs(root):
    results = list()
    for name in os.listdir(root):
        path = join(root, name)
        if os.path.isdir(path):
            results.append(path)
    return results


def dict_to_str(dictionary):
    return [f"{key}={val}" for key, val in dictionary.items()]


def get_path(root, **info):
    strings = dict_to_str(info)
    path = os.path.join(root, *strings)
    print(path)
