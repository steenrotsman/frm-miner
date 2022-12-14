import tensorflow as tf

ts = tf.constant([
    [0, 1, 2, 2, 1, 0],
    [0, 0, 1, 1, 0, 0],
    [2, 1, 1, 0, 0, 0],
    [2, 1, 0, 0, 1, 2],
    [0, 1, 0, 1, 0, 1]
], dtype=tf.float32)

rag = tf.ragged.constant([
    [1, 2],
    [3, 4, 5],
    [6, 7, 8, 9]
], dtype=tf.float32)
