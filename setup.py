#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="boa",
    version="0.0.1",
    description="Basis Overlap Architecture (BOA)",
    author="Manuel V. Klockow",
    author_email="manuel.klockow@iwr.uni-heidelberg.de",
    url="https://github.com/ManuelHei/boa",
    install_requires=[],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    # entry_points={
    #     "console_scripts": [
    #         "train_command = boa.train:main",
    #         "eval_command = src.eval:main",
    #     ]
    # },
)
