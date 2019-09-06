# -*- coding: utf-8 -*-
"""
Created on  06 2019
author: dylan

project: sciutils
version: 1.0
"""
import os
import re
import shutil
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


def oftype(path, types=None):
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


class Path(str):

    def __new__(cls, *paths, init=False, resolve=False):
        if len(paths) == 0:
            path = os.getcwd()
        else:
            path = os.path.join(*paths)
        if resolve:
            path = os.path.abspath(path)
        return super().__new__(cls, path)

    def __init__(self, *args, init=False, resolve=False):
        if init:
            self.init()

    def init(self):
        if not self.exists:
            if not self.ext:
                self.makedirs()
            else:
                with self.open("w") as f:
                    f.write("")

    @classmethod
    def here(cls, file):
        return cls(os.path.dirname(os.path.abspath(file)))

    @property
    def dirname(self):
        return self._new_subcls(os.path.dirname(self))

    @property
    def name(self):
        return os.path.basename(self)

    @property
    def basename(self):
        return os.path.splitext(self.name)[0]

    @property
    def ext(self):
        return os.path.splitext(self)[1]

    @property
    def is_file(self):
        return os.path.isfile(self)

    @property
    def is_dir(self):
        return os.path.isdir(self)

    @property
    def exists(self):
        return os.path.exists(self)

    @property
    def mtime(self):
        return os.path.getmtime(self)

    @property
    def ctime(self):
        return os.path.getctime(self)

    @property
    def stat(self):
        return os.stat(self)

    def _new_subcls(self, *args, **kwargs):
        return super().__new__(self.__class__, *args, **kwargs)

    def resolve(self):
        return self._new_subcls(os.path.abspath(self))

    def relpath(self, start):
        return Path(os.path.relpath(self, start))

    def join(self, *paths, init=False, resolve=False):
        return Path(self, *paths, init=init, resolve=resolve)

    def parts(self):
        path = str(self)
        parts = list()
        while len(path) and path != "/":
            path, part = os.path.split(path)
            parts.append(part)
        return parts[::-1]

    def makedirs(self, *subpaths):
        path = os.path.join(self, *subpaths)
        if not os.path.isdir(path):
            os.makedirs(path)
        return self._new_subcls(path)

    def rename(self, new_name, use_ext=True):
        if use_ext:
            new_name = new_name + self.ext
        new_path = os.path.join(self.dirname, new_name)
        os.rename(self, new_path)
        return self._new_subcls(new_path)

    def move(self, *dst):
        new_path = os.path.join(*dst, self.name)
        shutil.move(self, new_path)
        return self._new_subcls(new_path)

    def open(self, mode, *args, **kwargs):
        return open(self, mode, *args, **kwargs)

    def listdir(self):
        res = list()
        for name in os.listdir(self):
            path = os.path.join(self, name)
            res.append(self._new_subcls(path))
        return res

    def walk(self):
        res = list()
        for root, dirs, files in os.walk(self):
            for name in (list(dirs) + list(files)):
                res.append(self._new_subcls(os.path.join(root, name)))
        return res

    def files(self, *types, deep=True):
        res = list()
        for root, _, files in os.walk(self):
            for name in files:
                path = Path(os.path.join(root, name))
                if types and path.ext in types:
                    res.append(path)
                elif not types:
                    res.append(path)
            if not deep:
                break
        return res

    def dirs(self, deep=True):
        res = list()
        for root, dirs, _ in os.walk(self):
            for name in dirs:
                res.append(self._new_subcls(os.path.join(root, name)))
            if not deep:
                break
        return res

    # =========================================================================

    def findall(self, pattern):
        return re.findall(pattern, self)

    def re_search(self, pattern):
        return re.search(pattern, self)

    def re_search_number(self, keyword):
        return re.search(keyword + "(\d+)", self)

    # =========================================================================

    def search(self, *txts, deep=True):
        paths = self.walk() if deep else self.listdir()
        pattern = "|".join(list(txts))
        res = list()
        for path in paths:
            if len(path.findall(pattern)):
                res.append(Path(path))
        return res

    def search_files(self, *txts, deep=True):
        return [path for path in self.search(*txts, deep=deep) if path.is_file]


# class Folder(str):
#
#     """ Legacy support """
#
#     def __new__(cls, *paths):
#         if len(paths) == 0:
#             paths = [os.getcwd()]
#         return super().__new__(cls, os.path.join(*paths))
#
#     @classmethod
#     def here(cls, file):
#         return cls(os.path.dirname(os.path.abspath(file)))
#
#     @property
#     def dir(self):
#         return os.path.dirname(self)
#
#     @property
#     def name(self):
#         return os.path.split(self)[1]
#
#     @property
#     def mtime(self):
#         return os.path.getmtime(self)
#
#     @property
#     def ctime(self):
#         return os.path.getctime(self)
#
#     @staticmethod
#     def oftype(path, types):
#         return oftype(path, types)
#
#     def list_files(self, types=None):
#         types = _format_types(types)
#         res = list()
#         for name in os.listdir(self):
#             path = os.path.join(self, name)
#             if os.path.isfile(path) and oftype(path, types):
#                 res.append(File(path))
#         return res
#
#     def walk_files(self, types=None):
#         types = _format_types(types)
#         res = list()
#         for root, _, files in os.walk(self):
#             for path in [os.path.join(root, name) for name in files]:
#                 if oftype(path, types):
#                     res.append(File(path))
#         return res
#
#     def list_dirs(self):
#         res = list()
#         for name in os.listdir(self):
#             path = os.path.join(self, name)
#             if os.path.isdir(path):
#                 res.append(Folder(path))
#         return res
#
#     def walk_dirs(self):
#         res = list()
#         for root, dirs, _ in os.walk(self):
#             for path in [os.path.join(root, name) for name in dirs]:
#                 res.append(Folder(path))
#         return res
#
#     def makedirs(self, *subpaths):
#         path = os.path.join(self, *subpaths)
#         if not os.path.isdir(path):
#             os.makedirs(path)
#         return Folder(path)
#
#     def subfolder(self, *subpaths):
#         path = os.path.join(self, *subpaths)
#         return self.makedirs(path)
#
#     def find(self, *txts, deep=True):
#         """
#
#         Parameters
#         ----------
#         txts: array_like of str
#             Strings to search for
#         deep: bool, default: True
#             Search in subfolders
#         Returns
#         -------
#         paths: list of str
#         """
#         paths = list()
#         if not txts:
#             return self.walk_files if deep else self.list_files()
#         for path in self.walk_files():
#             name = os.path.split(path)[1]
#             if all([x in name for x in txts]):
#                 paths.append(path)
#         return paths
#
#     def __str__(self):
#         indent = "   "
#         string = "Folder: " + self
#         for name in self.list_dirs():
#             string += f"\n{indent}{name}"
#         for name in self.list_files():
#             string += f"\n{indent}{name}"
#         return string
#
#
# class File(str):match
#
#     """ Legacy support """
#
#     def __new__(cls, path):
#         return super().__new__(cls, path)
#
#     @property
#     def dir(self):
#         return os.path.split(self)[0]
#
#     @property
#     def name(self):
#         return os.path.split(self)[1]
#
#     @property
#     def ext(self):
#         return os.path.splitext(self)[1]
#
#     @property
#     def mtime(self):
#         return os.path.getmtime(self)
#
#     @property
#     def ctime(self):
#         return os.path.getctime(self)
#
#     def rename(self, new_name):
#         new_path = os.path.join(self.dir, new_name)
#         os.rename(self, new_path)
#         return Folder(new_path)
#
#     def move(self, dst):
#         new_path = os.path.join(dst, self.name)
#         shutil.move(self, new_path)
#         return Folder(new_path)


class Data(dict):

    def __init__(self, *paths):
        super().__init__()
        self.path = ""
        if paths:
            self.open(*paths)

    @property
    def filename(self):
        return self.path.filename

    @property
    def keylist(self):
        return list(self.keys())

    def save(self):
        np.savez(self.path, **self)

    def open(self, *paths):
        self.path = Path(*paths)
        if os.path.isfile(self.path):
            self._read()

    def _read(self):
        npzfile = np.load(self.path)
        for key, data in npzfile.items():
            super().update({key: data})
