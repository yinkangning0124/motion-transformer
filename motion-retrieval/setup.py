"""Installation script for the 'retrieval' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os

root_dir = os.path.dirname(os.path.realpath(__file__))


# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "joblib==1.2.0",
    "scann==1.2.8",
]

# Installation operation
setup(
    name="retrieval",
    author="Zhengbang",
    version="0.1.0",
    # description="Benchmark environments for high-speed robot learning in NVIDIA IsaacGym.",
    keywords=["retrieval", "motion intelligence"],
    include_package_data=True,
    python_requires=">=3.7.*",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    zip_safe=False,
)
