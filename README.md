# FRM-Miner #

Frequent Representative Motif Miner to discover frequently occurring motifs of variable length.
This repository contains the implementation of FRM-Miner as a Python package in the repository `frm`.
Code for the experiments and figures created for the test is contained in the repository `experiments`.

Code for experiment 5.1 is found in `experiments/benchmark.py`, it uses data from the [UCR Archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/).
Code for experiment 5.2 is found in `experiments/quantitative.py`, it uses simulated data.
Code for experiment 5.3 is found in `experiments/bike.py`, it uses data collected by the authors, which is provided in the repository `experiments/bike`.
Code for Figure 1 and Figure 2 is found in `experiments/qualitative.py`, it uses the [MPEG-7](https://dabi.temple.edu/external/shape/MPEG7/dataset.html) data set.


## Dependencies
### FRM-Miner dependencies
* NumPy

### Additional dependencies for experiments and figures
* Matplotlib
* opencv-python
* Pillow
* stumpy
* fitdecode (optional)