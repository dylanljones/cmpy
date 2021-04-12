# cmpy 0.0.5

NOTE: This project is still under development!

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
--------

| Module | Description  |
|:-------|:-----|
| basis.py | Tools for many-body basis representations  |
| matrix.py | Matrix tools and np.ndarray-wrapper  |
| operator.py | Abstract linear operator, sparse implementation and other tools |
| dos.py | Methods for computing the density of states |