from setuptools import setup, find_packages


def requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()


setup(
    name='cmpy',
    version='0.0.4',
    description='',
    url='https://github.com/dylanljones/cmpy',
    license='MIT',
    author='Dylan Jones',
    author_email='dylanljones94@gmail.com',
    packages=find_packages(),
    requirements=requirements(),
)
