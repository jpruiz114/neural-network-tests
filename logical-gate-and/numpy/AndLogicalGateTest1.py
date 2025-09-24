import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Training data (AND gate)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [0],
              [0],
              [1]])

# Initialize weights randomly
np.random.seed(42)
weights = 2 * np.random.random((2,1)) - 1
bias = np.random.random(1)

output = None

# Train for 10000 iterations
learning_rate = 0.1
for epoch in range(10000):
    # Forward pass
    z = np.dot(X, weights) + bias
    output = sigmoid(z)

    # Calculate error
    error = y - output

    # Backpropagation
    adjustments = error * sigmoid_derivative(output)
    weights += learning_rate * np.dot(X.T, adjustments)
    bias += learning_rate * np.sum(adjustments)

np.set_printoptions(precision=3, suppress=True)  # 3 decimals, no scientific notation

# Test results
print("Predictions after training:")
print(output)
