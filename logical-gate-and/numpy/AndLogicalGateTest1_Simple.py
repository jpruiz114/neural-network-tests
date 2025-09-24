import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Training data (AND gate)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [0],
              [0],
              [1]])

# Simple initialization - start close to solution
weights = np.array([[2.0], [2.0]])  # Positive weights
bias = np.array([-3.0])  # Negative bias (threshold)

learning_rate = 1.0
convergence_threshold = 0.01

print("Training AND Gate with simple approach:")
print("Epoch\tError\t\tConverged?")
print("-" * 40)

for epoch in range(500):  # Max 500 iterations
    # Forward pass
    z = np.dot(X, weights) + bias
    output = sigmoid(z)
    
    # Calculate error
    error = y - output
    total_error = np.mean(np.abs(error))
    
    # Print progress every 25 epochs
    if epoch % 25 == 0 or total_error < convergence_threshold:
        converged = "YES" if total_error < convergence_threshold else "NO"
        print(f"{epoch:4d}\t{total_error:.6f}\t{converged}")
        
        # Early stopping when converged
        if total_error < convergence_threshold:
            print(f"\nConverged after {epoch} iterations!")
            break
    
    # Simple gradient descent
    for i in range(len(X)):
        # Calculate gradient for this sample
        grad_weights = error[i] * X[i].reshape(-1, 1)
        grad_bias = error[i]
        
        # Update weights and bias
        weights += learning_rate * grad_weights
        bias += learning_rate * grad_bias

# Set display options
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
