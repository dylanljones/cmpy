# -*- coding: utf-8 -*-
"""
Created on  06 2019
author: dylan

project: sciutils
version: 1.0
"""
import os
import pickle
import numpy as np


def save_pkl(file, *args, info=None):
    data = list(args) + [info]
    with open(file, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(file):
    with open(file, "rb") as f:
        result = pickle.load(f)
    return result


def _oftype(path, types=None):
    if types is not None and os.path.splitext(path)[1] not in types:
        return False
    return True


def _format_types(types=None):
    if types is not None:
        if isinstance(types, str):
            types = [types]
        for i in range(len(types)):
            t = types[i]
            if not t.startswith("."):
                types[i] = "." + t
    return types


class File(str):

    def __new__(cls, path):
        return super().__new__(cls, path)

    @property
    def dir(self):
        return os.path.split(self)[0]

    @property
    def name(self):
        return os.path.split(self)[1]

    @property
    def ext(self):
        return os.path.splitext(self)[1]

    @property
    def mtime(self):
        return os.path.getmtime(self)

    @property
    def ctime(self):
        return os.path.getctime(self)

    def rename(self, new_name):
        new_path = os.path.join(self.dir, new_name)
        os.rename(self, new_path)
        return Folder(new_path)

    def move(self, dst):
        new_path = os.path.join(dst, self.name)
        shutil.move(self, new_path)
        return Folder(new_path)


class Folder(str):

    def __new__(cls, path):
        return super().__new__(cls, path)

    @classmethod
    def here(cls, file):
        return cls(os.path.dirname(os.path.abspath(file)))

    @property
    def dir(self):
        return os.path.dirname(self)

    @property
    def name(self):
        return os.path.split(self)[1]

    @property
    def mtime(self):
        return os.path.getmtime(self)

    @property
    def ctime(self):
        return os.path.getctime(self)

    @staticmethod
    def oftype(path, types):
        return _oftype(path, types)

    def list_files(self, types=None):
        types = _format_types(types)
        res = list()
        for name in os.listdir(self):
            path = os.path.join(self, name)
            if os.path.isfile(path) and _oftype(path, types):
                res.append(File(path))
        return res

    def walk_files(self, types=None):
        types = _format_types(types)
        res = list()
        for root, _, files in os.walk(self):
            for path in [os.path.join(root, name) for name in files]:
                if _oftype(path, types):
                    res.append(File(path))
        return res

    def list_dirs(self):
        res = list()
        for name in os.listdir(self):
            path = os.path.join(self, name)
            if os.path.isdir(path):
                res.append(Folder(path))
        return res

    def walk_dirs(self):
        res = list()
        for root, dirs, _ in os.walk(self):
            for path in [os.path.join(root, name) for name in dirs]:
                res.append(Folder(path))
        return res

    def makedirs(self, *subpaths):
        path = os.path.join(self, *subpaths)
        if not os.path.isdir(path):
            os.makedirs(path)
        return Folder(path)

    def subfolder(self, *subpaths):
        path = os.path.join(self, *subpaths)
        return self.makedirs(path)

    def find(self, *txts, deep=True):
        paths = list()
        if not txts:
            return self.walk_files if deep else self.files
        for path in self.walk_files():
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

    def __init__(self, *paths):
        super().__init__()
        self.path = ""
        if paths:
            self.open(os.path.join(*paths))

    @property
    def filename(self):
        fn = os.path.split(self.path)[1]
        return os.path.splitext(fn)[0]

    @property
    def keylist(self):
        return list(self.keys())

    def save(self):
        np.savez(self.path, **self)

    def open(self, *paths):
        self.path = os.path.join(*paths)
        if os.path.isfile(self.path):
            self._read()

    def _read(self):
        npzfile = np.load(self.path)
        for key, data in npzfile.items():
            super().update({key: data})
