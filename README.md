# cmpy 0.0.5

NOTE: This project is still under development and might change significantly!

`cmpy` is a collection of tools for condensed matter computational physics.

## Installation

Download package and install via pip
````commandline
pip install -e <folder path>
````
or the setup.py script
````commandline
python setup.py install
````

## Contents


### Main modules

| Module | Description  |
|:-------|:-----|
| hdf5 | Hdf5-database tools |
| basis | Tools for many-body basis representations  |
| matrix | Matrix tools and np.ndarray-wrapper  |
| operator | Abstract linear operator, sparse implementation and other tools |
| dos | Methods for computing the density of states |
| collection | Collection of random functions and constants |

### Models

Collection of common condensed matter models (unstable, might change significantly)

| Module | Description | Lattice support |
|:-------|:-----|:-------|
| abc | Model-Parameter container and abstract base classes  |  - |
| anderson | Anderson imurity models | no |
| ising | Ising model | yes |
| hubbard | Hubbard model | no |
| tightbinding | Thight-Binding model | yes |


## Usage

#### Basis

A ``Basis`` object can be initalized with the number of sites in the (many-body) system:

````python
from cmpy import Basis
basis = Basis(num_sites=3)
````

The corresponding states of a particle sector can be obtained by calling:
````python
sector = basis.get_sector(n_up=1, n_dn=1)
````
If no filling for a spin-sector is passed all possible fillings are included.
The labels of all states in a sector can be created by the ``state_labels`` method:
````python
>>> sector.state_labels()
['..⇅', '.↓↑', '↓.↑', '.↑↓', '.⇅.', '↓↑.', '↑.↓', '↑↓.', '⇅..']
````
The states of a sector can be iterated by the ``states``-property.
Each state consists of an up- and down-``SpinState``:
````python
state = list(sector.states)[0]
up_state = state.up
dn_state = state.dn
````
Each ``SpinState`` provides methods for optaining information about the state, for example:
`````python
>>> up_state.binstr(width=3)
001
>>> up_state.n
1
>>> up_state.occupations()
[1]
>>> up_state.occ(0)
1
>>> up_state.occ(1)
0
>>> up_state.occ(2)
0
`````


#### Matrix

The ``matrix``-module provides usefull methods for dealing with matrices.
All methods can also be accessed through the ``Matrix``-object, which is a wrapper of
``np.ndarray``:

````python
from cmpy import Matrix

mat = Matrix.zeros(3, 3)
````

#### Operators

The ``operators``-module provides the base-class ``LinearOperator`` based on ``scipy.LinearOperator``.
A simple sparse implementation which supports appending data is also included:
````python
import numpy as np
from cmpy import SparseOperator

shape = (3, 3)
op = SparseOperator(shape)
op.append(0, 0, 1)
op.append(1, 1, 1)
op.append(2, 2, 1)

op.matvec(np.ones(3))
````

