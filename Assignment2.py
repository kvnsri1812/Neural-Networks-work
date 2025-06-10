import numpy as np
import tensorflow as tf
import cv2
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

 
# Q1: Convolution Operations
 

# Define 5x5 input matrix
input_matrix = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
], dtype=np.float32).reshape((1, 5, 5, 1))  # Shape: [batch, height, width, channels]

# Define 3x3 kernel
kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=np.float32).reshape((3, 3, 1, 1))  # Shape: [filter_height, filter_width, in_channels, out_channels]

# Function to perform convolution with specified stride and padding
def apply_convolution(input_tensor, kernel_tensor, stride, padding):
    return tf.nn.conv2d(input_tensor, kernel_tensor, strides=[1, stride, stride, 1], padding=padding).numpy().squeeze()

# Apply and print convolutions with all parameter combinations
print("  Q1: Convolution Outputs  ")
print("Stride=1, Padding='VALID':\n", apply_convolution(input_matrix, kernel, 1, 'VALID'))
print("Stride=1, Padding='SAME':\n", apply_convolution(input_matrix, kernel, 1, 'SAME'))
print("Stride=2, Padding='VALID':\n", apply_convolution(input_matrix, kernel, 2, 'VALID'))
print("Stride=2, Padding='SAME':\n", apply_convolution(input_matrix, kernel, 2, 'SAME'))

 
# Q2 Task 1: Sobel Edge Detection
 

# Create a 128x128 grayscale image with random pixel values
image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)

# Define Sobel-X and Sobel-Y kernels manually
sobel_x_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y_kernel = np.array([
    [-1, -2, -1],
    [0,  0,  0],
    [1,  2,  1]
], dtype=np.float32)

# Apply Sobel filters manually using OpenCV's filter2D
sobel_x = cv2.filter2D(image, -1, sobel_x_kernel)
sobel_y = cv2.filter2D(image, -1, sobel_y_kernel)

# Display images
cv2.imshow("Original Image", image)
cv2.imshow("Sobel - X Direction", sobel_x)
cv2.imshow("Sobel - Y Direction", sobel_y)
cv2.waitKey(0)
cv2.destroyAllWindows()

 
# Q2 Task 2: Pooling Operations
 

# Create random 4x4 matrix as image
input_pool = tf.constant(np.random.rand(1, 4, 4, 1), dtype=tf.float32)

# Apply Max and Average Pooling
max_pool = tf.nn.max_pool2d(input_pool, ksize=2, strides=2, padding='VALID')
avg_pool = tf.nn.avg_pool2d(input_pool, ksize=2, strides=2, padding='VALID')

# Print pooling results
print("\n  Q2: Pooling  ")
print("Original Matrix:\n", input_pool.numpy().squeeze())
print("Max Pooling:\n", max_pool.numpy().squeeze())
print("Average Pooling:\n", avg_pool.numpy().squeeze())

 
# Q3: Standardization vs Normalization
 

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Normalize (Min-Max)
X_minmax = MinMaxScaler().fit_transform(X)

# Standardize (Z-score)
X_zscore = StandardScaler().fit_transform(X)

# Function to evaluate Logistic Regression on given data
def evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

# Get accuracy for original, normalized and standardized
acc_orig = evaluate_model(X, y)
acc_norm = evaluate_model(X_minmax, y)
acc_std = evaluate_model(X_zscore, y)

# Print results
print("\n  Q3: Accuracy Comparison  ")
print(f"Original Data Accuracy: {acc_orig:.4f}")
print(f"Min-Max Normalized Accuracy: {acc_norm:.4f}")
print(f"Z-score Standardized Accuracy: {acc_std:.4f}")

# Plot distributions
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.hist(X.flatten(), bins=30, color='blue')
plt.title("Original Features")

plt.subplot(1, 3, 2)
plt.hist(X_minmax.flatten(), bins=30, color='green')
plt.title("Min-Max Normalized")

plt.subplot(1, 3, 3)
plt.hist(X_zscore.flatten(), bins=30, color='red')
plt.title("Z-score Standardized")
plt.tight_layout()
plt.show()
