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

# Better initialization - start closer to solution
np.random.seed(42)
weights = np.array([[2.0], [2.0]])  # Start with positive weights
bias = np.array([-3.0])  # Start with negative bias

# Optimized parameters
learning_rate = 1.0
convergence_threshold = 0.01

print("Training AND Gate (Optimized):")
print("Epoch\tError\t\tConverged?")
print("-" * 40)

for epoch in range(1000):  # Max 1000 iterations
    # Forward pass
    z = np.dot(X, weights) + bias
    output = sigmoid(z)
    
    # Calculate error
    error = y - output
    total_error = np.mean(np.abs(error))
    
    # Print progress every 100 epochs
    if epoch % 100 == 0 or total_error < convergence_threshold:
        converged = "YES" if total_error < convergence_threshold else "NO"
        print(f"{epoch:4d}\t{total_error:.6f}\t{converged}")
        
        # Early stopping when converged
        if total_error < convergence_threshold:
            print(f"\nConverged after {epoch} iterations!")
            break
    
    # Backpropagation
    adjustments = error * sigmoid_derivative(output)
    weights += learning_rate * np.dot(X.T, adjustments)
    bias += learning_rate * np.sum(adjustments)

np.set_printoptions(precision=4, suppress=True)

print(f"\nFinal Results:")
print(f"Final error: {np.mean(np.abs(y - output)):.6f}")
print(f"\nPredictions vs Expected:")
print(f"Input\t\tPredicted\tExpected")
print("-" * 35)
for i in range(len(X)):
    print(f"{X[i]}\t\t{output[i][0]:.4f}\t\t{y[i][0]}")

print(f"\nFinal weights: {weights.flatten()}")
print(f"Final bias: {bias[0]:.4f}")