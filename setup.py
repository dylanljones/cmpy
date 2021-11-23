# coding: utf-8
#
# This code is part of cmpy.
#
# Copyright (c) 2020, Dylan Jones
#
# This code is licensed under the MIT License. The copyright notice in the
# LICENSE file in the root directory and this permission notice shall
# be included in all copies or substantial portions of the Software.

from setuptools import setup, find_packages


def requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()


def long_description():
    with open("README.md", "r") as f:
        return f.read()


setup(
    name='cmpy',
    version='0.0.6',
    author='Dylan Jones',
    author_email='dylanljones94@gmail.com',
    description='Collection of tools for condensed matter computational physics.',
    long_description=long_description(),
    long_description_content_type="text/markdown",
    url='https://github.com/dylanljones/cmpy',
    license='MIT License',
    packages=find_packages(),
    install_requires=requirements(),
    extras_require={
        'models': ["lattpy"]
    },
    python_requires='>=3.6',
)
