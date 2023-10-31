#!/bin/bash
pip install pybind11 optuna
git submodule update --init --recursive
bash compile.sh
python parameter_search.py