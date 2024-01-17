import numpy as np
from scipy.stats import zscore

ts = [
    [0, 1, 2, 2, 1, 0],
    [0, 0, 1, 1, 0, 0],
    [2, 1, 1, 0, 0, 0],
    [2, 1, 0, 0, 1, 2],
    [0, 1, 0, 1, 0, 1],
]

norm = [
    [-1.22474, 0, 1.22474, 1.22474, 0, -1.22474],
    [-0.707107, -0.707107, 1.41421, 1.41421, -0.707107, -0.707107],
    [1.78885, 0.447214, 0.447214, -0.894427, -0.894427, -0.894427],
    [1.22474, 0, -1.22474, -1.22474, 0, 1.22474],
    [-1, 1, -1, 1, -1, 1],
]

seq_1 = [
    'abccba',
    'aaccaa',
    'cccaaa',
    'cbaabc',
    'acacac',
]

seq_2 = [
    'aca',
    'aca',
    'cba',
    'cac',
    'bbb',
]

rag = [
    [1, 2],
    [3, 4, 5],
    [6, 7, 8, 9],
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9, 0],
]

rseq_1 = [
    'ac',
    'abc',
    'aacc',
    'abc',
    'abc',
    'bcca',
]

rseq_2 = [
    'b',
    'ac',
    'ac',
    'ac',
    'ac',
    'bb',
]

np.random.seed(0)
data = [zscore(np.random.random(np.random.randint(10, 1000))).tolist() for _ in range(100)]
