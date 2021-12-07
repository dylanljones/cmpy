# cmpy

![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/dylanljones/cmpy)
![GitHub](https://img.shields.io/github/license/dylanljones/cmpy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Collection of tools for condensed matter physics written in Python.

*NOTE*: This project is still under development and might contain errors or change significantly in the future!


## Installation

Install via `pip` from github:
```commandline
pip install git+git://github.com/dylanljones/cmpy.git@VERSION
```

or download/clone the package, navigate to the root directory and install via
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
| basis | Tools for many-body basis representations  |
| collection | Collection of helpful methods and constants |
| cpa | Coherent potantial approximation (on-site disorder) |
| disorder | Methods for constructing disorder | 
| exactdiag | Exact diagonalization methods |
| greens | Some implementations for the computation of Green's functions |
| matrix | Matrix tools and np.ndarray-wrapper  |
| operators | Abstract linear operator, sparse implementation and other tools |


### Models

Collection of common condensed matter models (unstable, might change significantly)

| Module | Description | Lattice support |
|:-------|:-----|:-------|
| abc | Model-Parameter container and abstract base classes  |  - |
| anderson | Anderson imurity models | :x: |
| ising | Ising model | :heavy_check_mark: |
| heisenberg | Heisenberg model | :heavy_check_mark: |
| hubbard | Hubbard model | :x: |
| tightbinding | Thight-Binding model | :heavy_check_mark: |


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
A simple sparse implementation of a Hamiltonian is also included.
````python
from cmpy import HamiltonOperator

size = 5
rows = [0, 1, 2, 3, 4, 0, 1, 2, 3, 1, 2, 3, 4]
cols = [0, 1, 2, 3, 4, 1, 2, 3, 4, 0, 1, 2, 3]
data = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
indices = (rows, cols)
hamop = HamiltonOperator(size, data, indices)
````
Converting the operator to an array yields
````python
>>> hamop.array()
[[0 1 0 0 0]
 [1 0 1 0 0]
 [0 1 0 1 0]
 [0 0 1 0 1]
 [0 0 0 1 0]]
````

The inlcuded models provide the method `hamilton_operator` to generate the 
`HamiltonOperator` for a specific particle sector or the full Hilber space, for example:
```python
>>> from cmpy.models import SingleImpurityAndersonModel

>>> siam = SingleImpurityAndersonModel(u=2, mu=None)    # Half filling
>>> hamop = siam.hamilton_operator(1, 1)                # Hamiltonian of sector 1, 1
HamiltonOperator(shape: (4, 4), dtype: float64)
```
