from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(
    name='cmpy',
    license="MIT",
    author='Dylan Jones',
    packages=["cmpy", "cmpy.core", "cmpy.hubbard", "cmpy.tightbinding"],
    install_requires=["numpy", "scipy", "matplotlib"],
)