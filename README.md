# FRM-Miner #

Frequent Representative Motif Miner (FRM-Miner): Efficiently Mining Frequent Representative Motifs in Large Collections of Time Series.

![Pipeline of FRM-Miner: A time series database is discretised into a sequence database using SAX. Non-overlapping frequent sequential patterns without gaps are mined, after which their occurrences are mapped back to the time series. The occurrences are then used to construct frequent representative motifs.](pipeline.png)

This repository contains the implementation of FRM-Miner as a Python package. By default, the C++ version is built and installed, but a pure Python implementation is provided as well.

# Installation

It is easiest to install FRM-Miner via pip:

```bash
pip install frm-miner
```

A C++ compiler is needed for this.
The C++ implementation can then be imported with `from frm import Miner`, the pure Python version can be imported with `from frm._frm_py.miner import Miner`.

# Example
You will probably get more meaningful results than this if you use your own data (collection of univariate time series, time series do not have to be equal length).
```
import numpy as np
import matplotlib.pyplot as plt  # Not in requirements

from frm import Miner

# Set hyperparameters
MINSUP = 0.3
SEGLEN = 5
ALPHABET = 5
K = 4

# Generate 10 random time series with 100 observations each
rng = np.random.default_rng()
data = [rng.standard_normal(100) for _ in range(10)]

# Mine frequent representative motifs
miner = Miner(MINSUP, SEGLEN, ALPHABET, k=K)
motifs = miner.mine(data)

# Plot frequent representative motifs
fig, axs = plt.subplots(ncols=K, sharey='all', layout='compressed')
for motif, ax in zip(motifs, axs):
    ax.plot(motif.representative)
plt.show()
```

