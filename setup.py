#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import setuptools

install_requires = [
    'scikit-learn',
    'pandas',
    'matplotlib',
    'networkx',
]
tests_require = ['pytest']

setuptools.setup(
    name="anomaly_detection_robustness",
    author="Roel Bertens",
    author_email="roelbertens@godatadriven.com",
    description="Investigate the robustness of anomaly detection models",
    license="MIT License",
    packages=setuptools.find_packages(exclude=['data', 'notebooks', 'tests']),
    install_requires=install_requires,
    test_suite="tests",
    tests_require=tests_require,
    extras_require={'test': tests_require},
    version='0.0.1',
)
