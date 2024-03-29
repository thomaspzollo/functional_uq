# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="functional_uq",
    version="0.0.1",
    description="Distribution-free guarantees for statistical functionals of loss CDF",
    long_description=readme,
    author="Thomas Zollo, Jake Snell, Zhun Deng",
    author_email="tpz2105@columbia.edu",
    url="https://github.com/thomaspzollo/functional_uq",
    license=license,
    packages=find_packages(exclude=("tests")),
)
