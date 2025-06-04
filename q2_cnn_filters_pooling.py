"""
Question 2: CNN Feature Extraction with Filters & Pooling
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# ------------- Task 1: Sobel Edge Detection --------------------------------
def sobel_demo(img_path: str | Path) -> None:
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot load image: {img_path}")

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    titles = ('Original', 'Sobel-X', 'Sobel-Y')
    images = (gray, sobel_x, sobel_y)

    plt.figure(figsize=(10, 3))
    for i, (img, title) in enumerate(zip(images, titles), 1):
        plt.subplot(1, 3, i)
        plt.imshow(img, cmap='gray')
        plt.title(title); plt.axis('off')
    plt.tight_layout(); plt.show()

# ------------- Task 2: MaxPool vs. AvgPool ---------------------------------
def pooling_demo() -> None:
    rnd = tf.random.uniform((1, 4, 4, 1), minval=0, maxval=10, dtype=tf.int32)
    rnd_np = rnd.numpy().squeeze()

    max_pool = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
    avg_pool = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)

    max_out = max_pool(tf.cast(rnd, tf.float32)).numpy().squeeze()
    avg_out = avg_pool(tf.cast(rnd, tf.float32)).numpy().squeeze()

    print("Original 4×4 matrix:\n", rnd_np)
    print("\n2×2 Max-pooled:\n", max_out)
    print("\n2×2 Average-pooled:\n", avg_out)

if __name__ == '__main__':
    # Path to *any* local grayscale test image
    sobel_demo('gray.jpeg')
    pooling_demo()
