"""
Question 1: Convolution Operations with Different Parameters
"""

import numpy as np
import tensorflow as tf

# Step 1. 5×5 input matrix (feel free to change numbers for quick testing)   #

input_matrix = np.array([
    [1,  2,  3,  0,  1],
    [0,  1,  2,  3,  4],
    [5,  4,  3,  2,  1],
    [1,  2,  3,  4,  5],
    [0,  1,  0,  1,  0]
], dtype=np.float32)

# Add batch- and channel-dims so TensorFlow sees (N, H, W, C)
x = tf.constant(input_matrix)[tf.newaxis, ..., tf.newaxis]

# Step 2. 3×3 kernel                                                         #
kernel = np.array([
    [ 1,  0, -1],
    [ 1,  0, -1],
    [ 1,  0, -1]
], dtype=np.float32)
k = tf.constant(kernel)[..., tf.newaxis, tf.newaxis]      # shape (3,3,1,1)

def conv(stride: int, padding: str) -> np.ndarray:
    """Utility wrapper for tf.nn.conv2d that returns a 2-D NumPy array."""
    out = tf.nn.conv2d(x, k, strides=[1, stride, stride, 1], padding=padding)
    return out.numpy().squeeze()


# Step 3 & 4. Run four configurations and print                              #
configs = [
    (1, 'VALID'),
    (1, 'SAME'),
    (2, 'VALID'),
    (2, 'SAME'),
]

if __name__ == '__main__':
    for stride, pad in configs:
        print(f"\n--- Stride = {stride}, Padding = '{pad}' ---")
        print(conv(stride, pad))
