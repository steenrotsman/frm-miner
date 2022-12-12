import tensorflow as tf

ts = tf.constant([
    [0, 1, 2, 2, 1, 0],
    [0, 0, 1, 1, 0, 0],
    [2, 1, 1, 0, 0, 0],
    [2, 1, 0, 0, 1, 2],
    [0, 1, 0, 1, 0, 1]
], dtype=float)

seq = [
    'abccba',
    'aabbaa',
    'cbbaaa',
    'cbaabc',
    'ababab'
]

patterns = [
    'a', 'aa', 'ab',
    'b', 'ba', 'baa',
    'c', 'cb'
]