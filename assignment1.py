import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime

print("\n--- Part 1: Tensor Manipulations & Reshaping ---")

tensor = tf.random.uniform(shape=(4, 6))
print("Original Tensor:\n", tensor.numpy())
print("Rank:", tf.rank(tensor).numpy())
print("Shape:", tensor.shape)

reshaped = tf.reshape(tensor, (2, 3, 4))
print("\nReshaped Tensor (2, 3, 4):\n", reshaped.numpy())
transposed = tf.transpose(reshaped, perm=[1, 0, 2])
print("\nTransposed Tensor (3, 2, 4):\n", transposed.numpy())

small_tensor = tf.constant([[1.0, 2.0, 3.0, 4.0]])
broadcasted_result = transposed + small_tensor
print("\nBroadcasted Result:\n", broadcasted_result.numpy())

print("\n--- Part 2: Loss Functions & Comparison ---")

y_true = tf.constant([1.0, 0.0, 0.0])
y_pred1 = tf.constant([0.9, 0.05, 0.05])
y_pred2 = tf.constant([0.6, 0.2, 0.2])

mse1 = tf.keras.losses.MSE(y_true, y_pred1).numpy()
mse2 = tf.keras.losses.MSE(y_true, y_pred2).numpy()
cce = tf.keras.losses.CategoricalCrossentropy()
cce1 = cce(y_true, y_pred1).numpy()
cce2 = cce(y_true, y_pred2).numpy()

print(f"MSE (Prediction 1): {mse1:.4f}")
print(f"MSE (Prediction 2): {mse2:.4f}")
print(f"CCE (Prediction 1): {cce1:.4f}")
print(f"CCE (Prediction 2): {cce2:.4f}")

plt.bar(['MSE_1', 'MSE_2', 'CCE_1', 'CCE_2'], [mse1, mse2, cce1, cce2])
plt.title("Loss Function Comparison")
plt.ylabel("Loss Value")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n--- Part 3: MNIST Training with TensorBoard ---")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x_train, y_train, epochs=5,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback])

print(f"\nTensorBoard logs saved at: {log_dir}")
print("To view, run in terminal: tensorboard --logdir logs/fit")
