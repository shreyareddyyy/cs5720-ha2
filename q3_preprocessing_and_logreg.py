import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Original data (no transformation)
log_reg_orig = LogisticRegression(max_iter=200)
log_reg_orig.fit(X_train, y_train)
y_pred_orig = log_reg_orig.predict(X_test)
acc_orig = accuracy_score(y_test, y_pred_orig)

# 2. Min-Max Normalization
minmax = MinMaxScaler()
X_train_minmax = minmax.fit_transform(X_train)
X_test_minmax = minmax.transform(X_test)

log_reg_minmax = LogisticRegression(max_iter=200)
log_reg_minmax.fit(X_train_minmax, y_train)
y_pred_minmax = log_reg_minmax.predict(X_test_minmax)
acc_minmax = accuracy_score(y_test, y_pred_minmax)

# 3. Z-score Standardization
standard = StandardScaler()
X_train_std = standard.fit_transform(X_train)
X_test_std = standard.transform(X_test)

log_reg_std = LogisticRegression(max_iter=200)
log_reg_std.fit(X_train_std, y_train)
y_pred_std = log_reg_std.predict(X_test_std)
acc_std = accuracy_score(y_test, y_pred_std)

# Print accuracies
print(f"Original data accuracy: {acc_orig:.2f}")
print(f"MinMax normalized accuracy: {acc_minmax:.2f}")
print(f"Standardized accuracy: {acc_std:.2f}")

# Visualize distributions
plt.figure(figsize=(15, 5))

# Original data
plt.subplot(131)
plt.hist(X_train[:, 0], bins=20, alpha=0.5, label='Feature 1')
plt.hist(X_train[:, 1], bins=20, alpha=0.5, label='Feature 2')
plt.title('Original Data')
plt.legend()

# MinMax Normalized
plt.subplot(132)
plt.hist(X_train_minmax[:, 0], bins=20, alpha=0.5, label='Feature 1')
plt.hist(X_train_minmax[:, 1], bins=20, alpha=0.5, label='Feature 2')
plt.title('MinMax Normalized')
plt.legend()

# Standardized
plt.subplot(133)
plt.hist(X_train_std[:, 0], bins=20, alpha=0.5, label='Feature 1')
plt.hist(X_train_std[:, 1], bins=20, alpha=0.5, label='Feature 2')
plt.title('Standardized')
plt.legend()

plt.show()