# FRM-Miner #

Frequent Representative Motif Miner (FRM-Miner): Efficiently Mining Frequent Representative Motifs in Large Collections of Time Series.

![Pipeline of FRM-Miner: A time series database is discretised into a sequence database using SAX. Non-overlapping frequent sequential patterns without gaps are mined, after which their occurrences are mapped back to the time series. The occurrences are then used to construct frequent representative motifs.](pipeline.png)

This repository contains the Python implementation of FRM-Miner (in `frm`), as well as code to replicate the experiments (in `experiments`).

# Installation

It is easiest to install FRM-Miner via pip:

```bash
pip install frm-miner
```

# Frm-Miner 1.0
Looking for the conference version (S. J. Rotman, B. Cule and L. Feremans, "Efficiently Mining Frequent Representative Motifs in Large Collections of Time Series," 2023 IEEE International Conference on Big Data (BigData), Sorrento, Italy, 2023, pp. 66-75, doi: 10.1109/BigData59044.2023.10386145.)?

[FRM-Miner 1.0](https://github.com/steenrotsman/frm-miner/tree/1.0)

# Example
You will probably get more meaningful results than this if you use your own data (collection of univariate time series, time series do not have to be equal length).
```
import numpy as np
import matplotlib.pyplot as plt  # Not a dependency

from frm import Miner

# Set hyperparameters
MINSUP = 0.3
SEGLEN = 5
ALPHA = 4
K = 4

# Generate 10 random time series with 100 observations each
rng = np.random.default_rng()
data = [rng.standard_normal(100) for _ in range(10)]

# Mine frequent representative motifs
miner = Miner(MINSUP, SEGLEN, ALPHA, k=K)
motifs = miner.mine(data)

# Plot frequent representative motifs
fig, axs = plt.subplots(ncols=K, sharey='all', layout='constrained')
for motif, ax in zip(motifs, axs):
    ax.plot(motif.representative)
plt.show()
```

