# FRM-Miner #

Frequent Representative Motif Miner to discover frequently occurring motifs of variable length.
This repository contains the implementation of FRM-Miner as a Python package in the repository `frm/_frm_py`.
The repository `src` contains the C++ implementation and code to bind the classes to Python.
Code for the experiments and figures created for the paper is contained in the repository `experiments`.

Running `pip install .` from this folder compiles and installs FRM-Miner (a C++ compiler is needed for this).
The C++ implementation can then be imported with `from frm import Miner`, the pure Python version can be imported with `from frm._frm_py.miner import Miner`.

## Dependencies
### FRM-Miner dependencies
* pybind11
* NumPy

### Additional dependencies for experiments and figures
* Matplotlib
* opencv-python
* Pillow
* stumpy
* pyscamp
* mass_ts
* tqdm
* yfinance (optional)
* fitdecode (optional)