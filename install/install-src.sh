#!/bin/bash

# Install this package and its dependencies from source.

source activate py_stringsimjoin_test_env

# System dependencies
if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then conda install --yes gcc; fi
which gcc

# Package dependencies
pip install -r requirements.txt

# Build dependencies
pip install Cython

# Build C++ extensions (Cython)
python setup.py build_ext --inplace

# Install package
python setup.py install
