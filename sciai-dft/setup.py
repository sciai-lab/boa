#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="mldft",
    version="0.0.1",
    description="OF-DFT using machine learning",
    author="Hamprecht Lab",
    author_email="",
    url="https://github.com/sciai-lab/structures25",
    install_requires=[],
    packages=find_packages(),
    python_requires=">=3.8",
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "mldft_ks = mldft.datagen.kohn_sham_dataset:main",
            "mldft_labelgen = mldft.datagen.generate_labels_dataset:main",
            "mldft_train = mldft.ml.train:main",
            "mldft_denop = mldft.ofdft.run_density_optimization:main",
        ]
    },
)
