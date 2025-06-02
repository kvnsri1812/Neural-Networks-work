import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime

# Part 1: Tensor Manipulations & Reshaping

print("\n--- Part 1: Tensor Manipulations & Reshaping ---")

# Create a random tensor of shape (4, 6)
tensor = tf.random.uniform(shape=(4, 6))
print("Original Tensor:\n", tensor.numpy())

# Find its rank and shape
print("Rank:", tf.rank(tensor).numpy())
print("Shape:", tensor.shape)

# Reshape into (2, 3, 4)
reshaped = tf.reshape(tensor, (2, 3, 4))
print("\nReshaped Tensor (2, 3, 4):\n", reshaped.numpy())

# Transpose to (3, 2, 4)
transposed = tf.transpose(reshaped, perm=[1, 0, 2])
print("\nTransposed Tensor (3, 2, 4):\n", transposed.numpy())

# Broadcasting a tensor of shape (1, 4)
small_tensor = tf.constant([[1.0, 2.0, 3.0, 4.0]])
broadcasted_result = transposed + small_tensor
print("\nBroadcasted Result:\n", broadcasted_result.numpy())

# Broadcasting Explanation:
# TensorFlow broadcasts smaller tensors by automatically expanding dimensions to match larger tensors for element-wise operations.


# Part 2: Loss Functions & Hyperparameter Tuning

print("\n--- Part 2: Loss Functions & Comparison ---")

# Define true and predicted values
y_true = tf.constant([1.0, 0.0, 0.0])
y_pred1 = tf.constant([0.9, 0.05, 0.05])
y_pred2 = tf.constant([0.6, 0.2, 0.2])

# Compute Mean Squared Error
mse1 = tf.keras.losses.MSE(y_true, y_pred1).numpy()
mse2 = tf.keras.losses.MSE(y_true, y_pred2).numpy()

# Compute Categorical Cross-Entropy
cce = tf.keras.losses.CategoricalCrossentropy()
cce1 = cce(y_true, y_pred1).numpy()
cce2 = cce(y_true, y_pred2).numpy()

# Print loss values
print(f"MSE (Prediction 1): {mse1:.4f}")
print(f"MSE (Prediction 2): {mse2:.4f}")
print(f"CCE (Prediction 1): {cce1:.4f}")
print(f"CCE (Prediction 2): {cce2:.4f}")

# Plot bar chart
plt.bar(['MSE_1', 'MSE_2', 'CCE_1', 'CCE_2'], [mse1, mse2, cce1, cce2])
plt.title("Loss Function Comparison")
plt.ylabel("Loss Value")
plt.grid(True)
plt.tight_layout()
plt.show()


# Part 3: Neural Network + TensorBoard Logging

print("\n--- Part 3: MNIST Training with TensorBoard ---")

# Load MNIST data and normalize
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# TensorBoard log path
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
model.fit(x_train, y_train, epochs=5,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback])

print(f"\nTensorBoard logs saved at: {log_dir}")
print("To view, run in terminal: tensorboard --logdir logs/fit")
