CS5720 Home Assignment 1 - TensorFlow Deep Learning

 Course Info
- Course: CS5720 Neural Networks and Deep Learning
- University: University of Central Missouri
- Term: Summer 2025

Name: Komatlapalli Venakata Naga Sri 

Student ID: 700773763

Assignment Summary:

- This assignment demonstrates core concepts in TensorFlow:
  - Tensor creation, reshaping, and broadcasting
  - Comparison of Mean Squared Error (MSE) and Categorical Cross-Entropy (CCE)
  - Training a neural network on MNIST and visualizing training with TensorBoard

**Part 1: Tensor Manipulations**

- Created a random tensor of shape (4, 6)
- Retrieved and printed its rank and shape
- Reshaped the tensor to (2, 3, 4)
- Transposed the reshaped tensor to (3, 2, 4)
- Created a smaller tensor of shape (1, 4) and added it using broadcasting
- Showcased how TensorFlow automatically expands dimensions during element-wise operations

**Part 2: Loss Function Comparison**

- Defined true labels as a one-hot vector [1, 0, 0]
- Created two sets of model predictions with different confidence levels
- Calculated MSE and CCE for both predictions
- Displayed results as printed output and bar chart using Matplotlib
- Observed how CCE is more sensitive to incorrect confident predictions than MSE

**Part 3: Neural Network and TensorBoard**

- Loaded and normalized the MNIST dataset
- Constructed a simple neural network:
  - Flatten input layer
  - Dense layer with 128 neurons (ReLU)
  - Output layer with 10 neurons (Softmax)
- Trained the model for 5 epochs with validation
- Logged training and validation metrics to the logs/fit directory
- Used TensorBoard to visualize accuracy and loss curves
- Observed overfitting trends and performance improvement across epochs


How to Run the Code:

- Install dependencies:
  pip install tensorflow matplotlib

- Run the script:
  python assignment.py

- Launch TensorBoard:
  tensorboard --logdir logs/fit


**4.1 Questions to Answer**:

Q1. What patterns do you observe in the training and validation accuracy curves?
   - Training accuracy increases steadily, showing that the model is learning.
   - Validation accuracy increases early, but may flatten or fluctuate after some epochs.
   - A consistent gap between training and validation may indicate overfitting.

Q2. How can you use TensorBoard to detect overfitting?
   - Overfitting is seen when:
   - Training loss decreases, but
   - Validation loss starts to increase
   - The accuracy gap widens between training and validation

Q3. What happens when you increase the number of epochs?
   - Initially, both accuracies improve.
   - After some point, validation accuracy plateaus while training accuracy continues increasing, indicating overfitting.
   - Longer training requires techniques like early stopping or dropout to avoid overfitting.


Repository Contents:

- assignment.py: Python script implementing all tasks
- README.md: This documentation file
- logs/fit/: TensorBoard log files (after training)
