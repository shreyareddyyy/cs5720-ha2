# CS5720 - Neural Networks and Deep Learning  
## University of Central Missouri  
*Course:* CS5720 Neural Networks and Deep Learning   
*Assignment:* Home Assignment 2  
*Term:* Summer 2025  
*Student Name:* shreya surakanti  

---

## Overview

This repository contains my submission for Home Assignment 2. It includes implementations for convolution operations, CNN feature extraction (using filters and pooling), and data preprocessing techniques using Python and relevant libraries such as TensorFlow, NumPy, OpenCV, and scikit-learn.

---

## Contents

- *Q1 - Convolution with Different Parameters*
- *Q2 - CNN Feature Extraction using Filters and Pooling*
- *Q3 - Data Preprocessing (Normalization vs. Standardization + Logistic Regression)*
- requirements.txt – Dependencies to install
- README.md – This file
- sample_grayscale.jpg – Used for Sobel filtering (or generated if missing)

---

## Question 1: Convolution Operations with Different Parameters

This task demonstrates how convolution behaves with different combinations of:
- Stride = 1 or 2
- Padding = 'VALID' or 'SAME'

A 5×5 input matrix and a 3×3 kernel are used. TensorFlow is used to apply convolution using tf.nn.conv2d. The output feature maps from each case are printed:
- Stride 1, Padding VALID
- Stride 1, Padding SAME
- Stride 2, Padding VALID
- Stride 2, Padding SAME

This helps visualize how padding and stride affect output dimensions and feature extraction.

---

## Question 2: CNN Feature Extraction with Filters and Pooling

### ➤ Task 1: Edge Detection Using Sobel Filter
This part uses OpenCV to load a grayscale image and applies the *Sobel filter* in both the X and Y directions. The Sobel filters highlight edges by detecting intensity changes in the image:
- Sobel-X detects vertical edges
- Sobel-Y detects horizontal edges

Three images are displayed using OpenCV:
1. Original image
2. Sobel-X output
3. Sobel-Y output


### ➤ Task 2: Max Pooling and Average Pooling
This demonstrates pooling operations using TensorFlow. A 4×4 random matrix is created and both:
- *2×2 Max Pooling*
- *2×2 Average Pooling*

are applied. It shows how pooling reduces the spatial dimensions while retaining essential information.

---

## Question 3: Data Preprocessing – Standardization vs. Normalization

This task focuses on preparing data for machine learning models.

1. *Dataset Used:* Iris dataset from sklearn.datasets
2. *Transformations Applied:*
   - *Min-Max Normalization* using MinMaxScaler
   - *Z-score Standardization* using StandardScaler
3. *Visualization:* Histograms compare the distributions before and after transformation.
4. *Model Training:* A Logistic Regression model is trained on:
   - Raw data
   - Normalized data
   - Standardized data

*Objective:* To analyze how preprocessing affects model performance and understand when to use normalization vs. standardization.

---

## ▶ How to Run

1. *Install all dependencies*:
   ```bash
   pip install -r requirements.txt
2. *run scripts* 
   ```bash
   python q1_convolution.py
   python q2_cnn_filters_pooling.py
   python q3_preprocessing_and_logreg.py
