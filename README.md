# cmpy

[![GitHub release (latest by date)][git-release]][repo-url]
[![Test][tests-master]][tests-url]
[![GitHub][license]][license-url]
[![Code style: black][black]][black-url]

Collection of tools for condensed matter physics written in Python.


| ❗  | Most of the code is beeing migrated to the [exactdiag] repository and will be removed soon! |
|----|:--------------------------------------------------------------------------------------------|


:warning: **WARNING**: This project is still under development and might contain errors or change significantly in the future!

## Installation

Install via `pip` from github:
```commandline
pip install git+https://github.com/dylanljones/cmpy.git@VERSION
```

or download/clone the package, navigate to the root directory and install via
````commandline
pip install .
````


## Contents


### Main modules

| Module     | Description                                                     |
|:-----------|:----------------------------------------------------------------|
| basis      | Tools for many-body basis representations                       |
| collection | Collection of helpful methods and constants                     |
| cpa        | Coherent potantial approximation (on-site disorder)             |
| disorder   | Methods for constructing disorder                               |
| exactdiag  | Exact diagonalization methods                                   |
| greens     | Some implementations for the computation of Green's functions   |
| matrix     | Matrix methods                                                  |
| operators  | Abstract linear operator, sparse implementation and other tools |


### Models

Collection of common condensed matter models (unstable, might change significantly)

| Module       | Description                                         | Lattice support    |
|:-------------|:----------------------------------------------------|:-------------------|
| abc          | Model-Parameter container and abstract base classes | -                  |
| anderson     | Anderson imurity models                             | :x:                |
| ising        | Ising model                                         | :heavy_check_mark: |
| heisenberg   | Heisenberg model                                    | :heavy_check_mark: |
| hubbard      | Hubbard model                                       | :x:                |
| tightbinding | Thight-Binding model                                | :heavy_check_mark: |


## Quick-Start

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

Many-Body Hamiltonian matrices can be constructed by projecting the
elements onto a basis sector. First, the basis and matrix array have to be initialized:
````python
import numpy as np
from cmpy import Basis

basis = Basis(num_sites=2)
sector = basis.get_sector()  # Full basis
ham = np.zeros((sector.size, sector.size))
````

The Hubbard Hamiltonian, for example, can then be constructed as follows:
````python
from cmpy import project_hubbard_inter, project_hopping

up_states = sector.up_states
dn_states = sector.dn_states

# Hubbard interaction
for i, j, val in project_hubbard_inter(up_states, dn_states, u=[2.0, 2.0]):
    ham[i, j] += val

# Hopping term
for i, j, val in project_hopping(up_states, dn_states, site1=0, site2=1, hop=1.0):
    ham[i, j] += val
````

[exactdiag]: https://github.com/dylanljones/exactdiag

[repo-url]: https://github.com/dylanljones/cmpy
[license-url]: https://github.com/dylanljones/cmpy/blob/master/LICENSE
[tests-url]: https://github.com/dylanljones/cmpy/actions/workflows/test.yml
[black-url]: https://github.com/psf/black

[git-release]: https://img.shields.io/github/v/release/dylanljones/cmpy?logo=Github&style=flat-square
[tests-master]: https://img.shields.io/github/workflow/status/dylanljones/cmpy/Test/master?label=tests&logo=github&style=flat-square
[license]: https://img.shields.io/github/license/dylanljones/cmpy?color=lightgrey&style=flat-square
[black]: https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square
