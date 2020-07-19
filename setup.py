from setuptools import setup, find_packages


def requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()


setup(
    name='cmpy',
    version='0.0.1',
    description='',
    url='',
    license='',
    author='Dylan Jones',
    author_email='',
    packages=find_packages(),
    requirements=requirements(),
)
