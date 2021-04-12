# cmpy 0.0.5

NOTE: This project is still under development and might change significantly!

`cmpy` is a collection of tools for condensed matter computational physics.

Installation
------------

Download package and install via pip
````commandline
pip install -e <folder path>
````
or the setup.py script
````commandline
python setup.py install
````

Contents
========

Main modules
------------

| Module | Description  |
|:-------|:-----|
| hdf5 | Hdf5-database tools |
| basis | Tools for many-body basis representations  |
| matrix | Matrix tools and np.ndarray-wrapper  |
| operator | Abstract linear operator, sparse implementation and other tools |
| dos | Methods for computing the density of states |
| collection | Collection of random functions and constants |


Models
------
Collection of common condensed matter models (unstable, might change significantly)

| Module | Description | Lattice support |
|:-------|:-----|:-------|
| abc | Model-Parameter container and abstract base classes  |  - |
| anderson | Anderson imurity models | no |
| ising | Ising model | yes |
| hubbard | Hubbard model | no |
| tightbinding | Thight-Binding model | yes |
