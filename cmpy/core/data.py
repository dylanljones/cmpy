# -*- coding: utf-8 -*-
"""
Created on 25 Apr 2019
@author: Dylan Jones

project: cmpy
version: 1.0

"""
import os
from os.path import dirname
import numpy as np

PROJECT_DIR = dirname(dirname(dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, "_data")
IMG_DIR = os.path.join(PROJECT_DIR, "_imgs")


class Folder:

    def __init__(self, *parts):
        self.path = os.path.join(*parts)
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

    @property
    def name(self):
        return os.path.split(self.path)[1]

    @property
    def files(self):
        return self.listfiles()

    def build_path(self, *rel_names):
        return os.path.join(self.path, *rel_names)

    def mtime(self):
        return os.path.getmtime(self.path)

    def listdirs(self, full=True):
        res = list()
        for name in os.listdir(self.path):
            path = os.path.join(self.path, name)
            if os.path.isdir(path):
                res.append(path if full else name)
        return res

    def subfolders(self):
        folders = list()
        for path in self.listdirs():
            folders.append(Folder(path))
        return folders

    def listfiles(self, full=True):
        res = list()
        for name in os.listdir(self.path):
            path = os.path.join(self.path, name)
            if os.path.isfile(path):
                res.append(path if full else name)
        return res

    def walk(self, full=True):
        for root, dirs, files in os.walk(self.path):
            if full:
                dirs = [os.path.join(root, name) for name in dirs]
                files = [os.path.join(root, name) for name in files]
            yield root, dirs, files

    def walk_files(self):
        files = list()
        for _, _, fpaths in self.walk():
            files += fpaths
        return files

    def find(self, *txts, deep=True):
        paths = list()
        if not txts:
            return self.walk_files if deep else self.files
        for _, _, files in self.walk():
            for path in files:
                name = os.path.split(path)[1]
                if all([x in name for x in txts]):
                    paths.append(path)
        return paths

    def __str__(self):
        indent = "   "
        string = "Folder: " + self.path
        for name in self.listdirs(full=False):
            string += f"\n{indent}{name}"
        for name in self.listfiles(full=False):
            string += f"\n{indent}{name}"
        return string


class Data(dict):

    def __init__(self, path=None):
        super().__init__()
        self.path = ""
        if path is not None:
            self.open(path)

    @property
    def filename(self):
        fn = os.path.split(self.path)[1]
        return os.path.splitext(fn)[0]

    @property
    def keylist(self):
        return list(self.keys())

    def save(self):
        np.savez(self.path, **self)

    def open(self, path):
        self.path = path
        if os.path.isfile(self.path):
            self._read()

    def _read(self):
        npzfile = np.load(self.path)
        for key, data in npzfile.items():
            super().update({key: data})


Folder(IMG_DIR)
