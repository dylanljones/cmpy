# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2021, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

"""Tools for handling hdf5-files."""

import os
import h5py
from abc import ABC
from typing import Union, Optional, List


def data_keys(hdf5object):
    def makelist(name):
        keys.append(name)

    keys = []
    hdf5object.visit(makelist)
    return keys

def dict_to_hdf5(data_dict, hdf5_filename, path=""):
    myfile = File(os.path.join(path, hdf5_filename), mode="w")
    dict_parents = [data_dict]
    parents = [myfile]
    while dict_parents:
        new_dict_parents = list()
        new_parents = list()
        for root, filegroup in zip(dict_parents, parents):
            for key, value in root.items():
                if isinstance(value, dict):
                    new_filegroup = filegroup.create_group(key)
                    new_dict_parents.append(root[key])
                    new_parents.append(new_filegroup)
                else:
                    filegroup.create_dataset(key, data=value)
        parents = new_parents
        dict_parents = new_dict_parents

def hdf5_to_dict(hdf5_filename, path=""):
    myfile = File(os.path.join(path, hdf5_filename), mode="r")
    links = data_keys(myfile)
    tree = {}
    for path in links:
        node = tree
        splitpath = path.split('/')
        for pathindex in range(len(splitpath)):
            if splitpath[pathindex]:
                part_path = os.path.join(*splitpath[:pathindex+1])
                if isinstance(myfile[part_path], Group):
                    node = node.setdefault(splitpath[pathindex], dict())
                else:
                    node = node.setdefault(splitpath[pathindex], myfile[part_path][()])
    return tree

def require_group(parent: Union[h5py.File, h5py.Group], name: str,
                  track_order: Optional[bool] = False) -> h5py.Group:
    """Returns the specified group if it exists or creates the group otherwise.

    Parameters
    ----------
    parent : h5py.File or h5py.Group
        The parent item.
    name : str
        The name of the group.
    track_order : bool, optional
        If ``True`` the order of the child-items is tracked. The default is ``False``.

    Returns
    -------
    group : h5py.Group
        The new or existing group.
    """
    if name in parent:
        return parent[name]
    return parent.create_group(name, track_order)


def check_attrs(item: Union[h5py.File, h5py.Group, h5py.Dataset],
                attrs: dict, mode: Optional[str] = "equals") -> bool:
    """Checks if the attributes of an hdf5-object match the given attributes.

    Parameters
    ----------
    item : h5py.File or h5py.Group or h5py.Database
        The attributes of the item are checked.
    attrs : dict
        The attributes for matching.
    mode : str, optional
        Mode for matching attributes. Valid modes are 'equals' and 'contains'.
        If the mode is 'equals', the dictionary of the item attributes has to be
        equal to the giben dictionary. If the mode is 'contains', the item dictionary
        can contain any number of values, but the values of the given dictionary have
        to be included. The default is 'equals'.

    Returns
    -------
    matches: bool
    """
    if mode == "contains":
        for key, val in attrs.items():
            if key not in item.attrs.keys() or item.attrs[key] != val:
                return False
        return True
    elif mode == "equals":
        return item.attrs == attrs
    else:
        modes = ["contains", "equals"]
        raise ValueError(f"Mode '{mode}' not supported! Valid modes: {modes}")


def find_by_attrs(parent: Union[h5py.File, h5py.Group], attrs: dict,
                  mode: Optional[str] = "contains") -> Union[h5py.Group, h5py.Dataset, None]:
    """Returns the first child of the parent with matching attributes.

    Parameters
    ----------
    parent : h5py.File or h5py.Group
        The parent item.
    attrs : dict
        The attributes for mathcing the child item.
    mode : str, optional
        Mode for matching attributes. Valid modes are 'equals' and 'contains'.

    Returns
    -------
    child: h5py.Group or h5py.Dataset or None
        The child item or None if no matching child was found.
    """
    for k in parent.keys():
        item = parent[k]
        if check_attrs(item, attrs, mode):
            return item
    return None


def list_groups(parent: Union[h5py.File, h5py.Group]) -> List[h5py.Group]:
    """Returns a list of all child-Groups of an h5py Group or File.

    Parameters
    ----------
    parent : h5py.File or h5py.Group
        The parent item.

    Returns
    -------
    groups : list of h5py.Group
        The child-Group of the parent Group.
    """
    return [parent[k] for k in parent.keys() if isinstance(parent[k], h5py.Group)]


def list_datasets(parent: Union[h5py.File, h5py.Group]) -> List[h5py.Dataset]:
    """Returns a list of all child-Datasets of an h5py Group or File.

    Parameters
    ----------
    parent : h5py.File or h5py.Group
        The parent item.

    Returns
    -------
    dsets : list of h5py.Dataset
        The child-Datasets of the parent Group.
    """
    return [parent[k] for k in parent.keys() if isinstance(parent[k], h5py.Dataset)]


def list_children(parent: Union[h5py.File, h5py.Group]):
    """Returns all child-Groups and -Datasets of an h5py Group or File.

    Parameters
    ----------
    parent : h5py.File or h5py.Group
        The parent item.

    Returns
    -------
    groups : list of h5py.Group
        The child-Group of the parent Group.
        dsets : list of h5py.Dataset
        The child-Datasets of the parent Group.
    """
    groups, datasets = list(), list()
    for key in parent.keys():
        item = parent[key]
        if isinstance(item, h5py.Dataset):
            datasets.append(item)
        else:
            groups.append(item)
    return groups, datasets


def walk(parent: Union[h5py.File, h5py.Group]):
    """Iterates over all children of an item by walking the tree top-down.

    Parameters
    ----------
    parent : h5py.File or h5py.Group
        The parent item.

    Yields
    ------
    root: h5py.Group
        The parent item of the following children.
    groups : list of h5py.Group
        The child-Groups of the root Group.
    datasets : list of h5py.Database
        The child-Datasets of the root Group.
    """
    parents = [parent]
    while parents:
        new_parents = list()
        for root in parents:
            groups, dsets = list_children(root)
            yield root, groups, dsets
            new_parents.extend(groups)
        parents = new_parents


def rename(item: Union[h5py.File, h5py.Group, h5py.Dataset], new_name: str) -> None:
    """Renames an hdf5-object in-place.

    Parameters
    ----------
    item : h5py.File or h5py.Group or h5py.Dataset
        The hdf5-object to rename.
    new_name : str
        The new name of the object.
    """
    parent = item.parent
    if new_name != os.path.split(item.name)[1]:
        parent.move(item.name, new_name)


def get_maxshape(shape, resize_axis=None):
    """Creates the maxshape tuple for a new dataset and sets the resizable axes to `None`.

    Parameters
    ----------
    shape : array_like
        The initial maxshape values.
    resize_axis : array_like, optional
        The resizable axis indices.

    Returns
    -------
    maxshape : tuple of int or None
    """
    if resize_axis is None:
        return shape
    if isinstance(resize_axis, int):
        resize_axis = [resize_axis]
    shape = list(shape)
    for axis in resize_axis:
        shape[axis] = None
    return shape


def next_name(parent: Union[h5py.File, h5py.Group], name: str,
              index_first: Optional[bool] = False, zfill: Optional[int] = 0) -> str:
    """Creates the next unique name.

    Parameters
    ----------
    parent : h5py.File or h5py.Group
        The parent item.
    name : str
        The name of the new object. If the name is allready contained in the parent item,
        an index will be added with the next available number.
    index_first : bool, optional
        If True, an index will also be added to the first occurance of the name.
    zfill : int, optional
        The number of digits used for the index strings.

    Returns
    -------
    name: str
        The unique name.
    """
    if name in parent:
        max_index = 0
        for key in parent.keys():
            try:
                index = int(key.split("_")[-1])
                max_index = max(max_index, index)
            except ValueError:
                pass
        name = name + "_" + str(max_index + 1).zfill(zfill)
    elif index_first:
        index = 0
        name = name + "_" + str(index).zfill(zfill)
    return name


def treestr(parent: Union[h5py.File, h5py.Group, h5py.Dataset],
            attrs: Optional[bool] = False,
            indent: Optional[int] = 4,
            _lvl: Optional[int] = 0) -> str:
    """Builds a string representation of the database tree.

    Parameters
    ----------
    parent : h5py.File or h5py.Group
        The parent item. This should be the root or file item.
    attrs : bool, optional
        Flag if attributes of the items are printed next to names. The default is `False`.
    indent : int, optional
        The number of characters used for indentations. The default is `4`.
    _lvl : int, optional
        The current indentation-level of the tree. This is only used for recursive calls and
        shouldn't be passed by the user.

    Returns
    -------
    string : str
        The string representing the database.
    """
    # Define style of branches
    vline = "│" + " " * (indent - 1)
    hline = "├" + "─" * (indent - 1) + " "

    # Build string for parent item
    if _lvl == 0:
        head = ""
        name = parent.filename if isinstance(parent, h5py.File) else parent.name
    else:
        head = vline * (_lvl - 1) + hline
        name = os.path.split(parent.name)[1]
    if not isinstance(parent, h5py.Dataset):
        name += f" ({len(parent)})"

    attrs_str = ""
    if attrs and parent.attrs:
        attrs_str = "  " + str(dict(parent.attrs))

    string = f"{head}{name}{attrs_str}\n"

    # Recursely call method for child-items
    if not isinstance(parent, h5py.Dataset):
        # Child-Groups
        for key in parent.keys():
            item = parent[key]
            if not isinstance(item, h5py.Dataset):
                string += treestr(item, attrs, indent, _lvl + 1)
        # Child-Datasets
        for key in parent.keys():
            item = parent[key]
            if isinstance(item, h5py.Dataset):
                string += treestr(item, attrs, indent, _lvl + 1)

    return string


# =========================================================================
# Object wrappers
# =========================================================================

class BaseclassError(ValueError):

    def __init__(self, cls, obj):
        super().__init__(f"Can't create '{cls.__name__}' from '{type(obj).__name__}' instance!")


# noinspection PyUnusedLocal, PyMissingConstructor, PyUnresolvedReferences
class HDF5File(h5py.File):
    """Wrapper of ``h5py.File``.

    Parameters
    ----------
    filename : str
        The path of the file.
    mode : str, optional
        The mode for opening the file.
    """

    def __init__(self, name, mode=None, **kwargs):
        super().__init__(name, mode=mode, **kwargs)


# noinspection PyMissingConstructor
class HDF5Group(h5py.Group):
    """Wrapper of ``h5py.Group``.

    The group-wrapper can either be initialized with an existing instance of ``h5py.Group``
    or by the parent and name of the group. If the parent-item is used and the group doesn't
    exist it will be created.

    Parameters
    ----------
    parent : h5py.File or h5py.Group, optional
        The parent item of the group. If intializing with the parent object the name of the
        group has to be passed.
    name : str, optional
        The name of the group. If intializing with the parent object the parent of the group
        has to be passed.
    obj : h5py.Group, optional
        The ``h5py.Group``-instance for initializing the group with an existing instance.
        If ``obj`` is passed the other parameters are ignored.
    **kwargs
        Keyword arguments for creating a group if it doesn't exist.
    """

    def __new__(cls, parent: Optional[h5py.Group] = None, name: Optional[str] = "",
                obj: Optional[h5py.Group] = None, **kwargs):
        # create instance
        if obj is None:
            obj = parent.create_group(name, **kwargs)

        if not isinstance(obj, h5py.Group):
            raise BaseclassError(cls, obj)
        self = super().__new__(cls)
        self.__dict__ = obj.__dict__
        return self

    def __init__(self, *args, **kwargs):
        pass


# noinspection PyMissingConstructor
class HDF5Dataset(h5py.Dataset):
    """Wrapper of ``h5py.Dataset``.

    The dataset-wrapper can either be initialized with an existing instance of ``h5py.Dataset``
    or by the parent and name of the Dataset. If the parent-item is used and the dataset doesn't
    exist it will be created.

    Parameters
    ----------
    parent : h5py.File or h5py.Group, optional
        The parent item of the dataset. If intializing with the parent object the name of the
        dataset has to be passed.
    name : str, optional
        The name of the dataset. If intializing with the parent object the parent of the dataset
        has to be passed.
    obj : h5py.Dataset, optional
        The ``h5py.Dataset``-instance for initializing the dataset with an existing instance.
        If ``obj`` is passed the other parameters are ignored.
    **kwargs
        Keyword arguments for creating a dataset if it doesn't exist.
    """

    def __new__(cls, parent: Optional[h5py.Group] = None, name: Optional[str] = "",
                obj: Optional[h5py.Group] = None, **kwargs):
        # create instance
        if obj is None:
            obj = parent.create_dataset(name, **kwargs)

        if not isinstance(obj, h5py.Dataset):
            raise BaseclassError(cls, obj)
        self = super().__new__(cls)
        self.__dict__ = obj.__dict__
        return self

    def __init__(self, *args, **kwargs):
        pass


# noinspection PyArgumentList, PyUnresolvedReferences
class GroupInterface(ABC):
    """Abstract class defining group-interface and additional functionality.

    The interface is needed to implement the group-behavior for the ``h5py.Group``- and
    ``h5py.File``-objects.

    Attributes
    ----------
    _group : HDF5Group
        The object used to wrap ``h5py.Group``. If None this class is used.
    _dataset : HDF5Dataset
        The object used to wrap ``h5py.Dataset``.
    """
    _group: HDF5Group = None
    _dataset: HDF5Dataset = HDF5Dataset

    @property
    def key(self):
        return os.path.split(self.name)[1]

    def _cast_group(self, obj):
        cls = self.__class__ if self._group is None else self._group
        return cls(obj=obj)

    def _cast_dataset(self, obj):
        return self._dataset(obj=obj)  # noqa

    def _cast(self, obj):
        if isinstance(obj, h5py.Group):
            return self._cast_group(obj)
        elif isinstance(obj, h5py.Dataset):
            return self._cast_dataset(obj)
        return obj

    def __getitem__(self, item):
        """Cast returned items to wrapper classes."""
        obj = super().__getitem__(item)
        return self._cast(obj)

    def create_group(self, name, track_order=None):
        obj = super().create_group(name, track_order)
        return self._cast_group(obj)

    def create_dataset(self, name, shape=None, dtype=None, data=None, **kwds):
        obj = super().create_dataset(name, shape, dtype, data, **kwds)
        return self._cast_dataset(obj)

    def require_group(self, name, track_order=False):
        if self.__contains__(name):
            return self.__getitem__(name)
        return self.create_group(name, track_order)

    def require_dataset(self, name, shape=None, dtype=None, data=None, **kwds):
        if self.__contains__(name):
            return self.__getitem__(name)
        return self.create_dataset(name, shape, dtype, data, **kwds)

    def walk(self):
        for parent, groups, dsets in walk(self):  # noqa
            parent = self._cast(parent)
            groups = [self._cast_group(obj) for obj in groups]
            dsets = [self._cast_dataset(obj) for obj in dsets]
            yield parent, groups, dsets

    def list_children(self):
        for obj in list_children(self):  # noqa
            yield self._cast(obj)

    def list_groups(self):
        for obj in list_groups(self):  # noqa
            yield self._cast_group(obj)

    def list_datasets(self):
        for obj in list_datasets(self):  # noqa
            yield self._cast_dataset(obj)

    def treestr(self):
        return treestr(self)  # noqa

    def __repr__(self):
        return f'{self.__class__.__name__} "{self.name}" ({self.__len__()} members)'


# =========================================================================
# Simple objects
# =========================================================================


class Dataset(HDF5Dataset):

    def __init__(self, parent: Optional[h5py.Group] = None, name: Optional[str] = "",
                 obj: Optional[h5py.Group] = None, **kwargs):
        HDF5Dataset.__init__(self, parent=parent, name=name, obj=obj, **kwargs)


class Group(GroupInterface, HDF5Group):

    def __init__(self, parent: Optional[h5py.Group] = None, name: Optional[str] = "",
                 obj: Optional[h5py.Group] = None, **kwargs):
        GroupInterface.__init__(self)
        HDF5Group.__init__(self, parent=parent, name=name, obj=obj, **kwargs)


class File(GroupInterface, HDF5File):

    _group = Group

    def __init__(self, filename, mode="a", **kwargs):
        GroupInterface.__init__(self)
        HDF5File.__init__(self, filename, mode, **kwargs)
