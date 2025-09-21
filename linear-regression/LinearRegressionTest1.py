import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
m = 100  # number of samples
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)

# Add bias term (intercept) to X
X_b = np.c_[np.ones((m, 1)), X]

# Initialize theta (parameters)
theta = np.random.randn(2, 1)

# Learning rate and number of iterations
learning_rate = 0.05
iterations = 1000

# Gradient Descent
for i in range(iterations):
    gradients = 1/m * X_b.T.dot(X_b.dot(theta) - y)
    theta -= learning_rate * gradients

# Predictions
y_pred = X_b.dot(theta)

# Plot the results
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red', linewidth=2)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression")
plt.show()

# Output the learned parameters
print("Learned parameters:", theta)
