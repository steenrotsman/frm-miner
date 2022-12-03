import numpy as np

ts = np.array([
    [0, 1, 2, 2, 1, 0],
    [0, 0, 1, 1, 0, 0],
    [2, 1, 1, 0, 0, 0],
    [2, 1, 0, 0, 1, 2],
    [0, 1, 0, 1, 0, 1]
])

seq = [
    'abccba',
    'aabbaa',
    'cbbaaa',
    'cbaabc',
    'ababab'
]

patterns = [
    'a', 'b', 'c',
    'aa', 'ab', 'ba',
    'baa'
]