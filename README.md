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
| basis | Tools for many-body basis representations  |
| matrix | Matrix tools and np.ndarray-wrapper  |
| operator | Abstract linear operator, sparse implementation and other tools |
| dos | Methods for computing the density of states |
| collection | Collection of random functions and constants |


Models
------
Collection of common condensed matter models (unstable, might change significantly)

| Module | Description | Many-body | Lattice support |
|:-------|:-----|:-------|:---------|
| abc | Model-Parameter container and abstract base classes  | - | - |
| anderson | Anderson imurity models | yes | no |
| ising | Ising model | no | yes |
| tightbinding | Thight-Binding model | no | yes |
| hubbard | Hubbard model | yes | no |
