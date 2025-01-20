import numpy as np


# Activation functions and their derivatives
def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


# Initialize parameters
def initialize_parameters(layer_dims):
    np.random.seed(42)
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))
    return parameters


# Forward propagation
def forward_propagation(X, parameters):
    cache = {"A0": X}
    L = len(parameters) // 2
    for l in range(1, L + 1):
        Z = np.dot(parameters[f"W{l}"], cache[f"A{l - 1}"]) + parameters[f"b{l}"]
        if l == L:  # Output layer (linear activation for regression)
            A = Z
        else:  # Hidden layers
            A = relu(Z)
        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = A
    return cache


# Compute cost
def compute_cost(A_last, Y):
    m = Y.shape[1]
    cost = np.sum((A_last - Y) ** 2) / (2 * m)  # Mean Squared Error
    return cost


# Backward propagation
def backward_propagation(Y, cache, parameters):
    grads = {}
    L = len(parameters) // 2
    m = Y.shape[1]

    # Output layer gradients
    dZ = cache[f"A{L}"] - Y
    grads[f"dW{L}"] = np.dot(dZ, cache[f"A{L - 1}"].T) / m
    grads[f"db{L}"] = np.sum(dZ, axis=1, keepdims=True) / m

    # Hidden layers gradients
    for l in range(L - 1, 0, -1):
        dA = np.dot(parameters[f"W{l + 1}"].T, dZ)
        dZ = dA * relu_derivative(cache[f"Z{l}"])
        grads[f"dW{l}"] = np.dot(dZ, cache[f"A{l - 1}"].T) / m
        grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True) / m

    return grads


# Update parameters
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]
    return parameters


# Train the model
def train(X, Y, layer_dims, learning_rate=0.01, epochs=1000):
    parameters = initialize_parameters(layer_dims)
    for epoch in range(epochs):
        # Forward propagation
        cache = forward_propagation(X, parameters)

        # Compute cost
        cost = compute_cost(cache[f"A{len(layer_dims) - 1}"], Y)

        # Backward propagation
        grads = backward_propagation(Y, cache, parameters)

        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost = {cost:.4f}")
    return parameters


# Predict
def predict(X, parameters):
    cache = forward_propagation(X, parameters)
    return cache[f"A{len(parameters) // 2}"]


# Example usage
if __name__ == "__main__":
    # Sample regression data
    np.random.seed(1)
    X = np.random.randn(1, 500)  # 1 feature, 500 examples
    Y = 3 * X + np.random.randn(1, 500) * 0.1  # Linear relation with noise

    # Define layer dimensions
    layer_dims = [1, 10, 1]  # 1 input feature, 10 hidden units, 1 output

    # Train the model
    parameters = train(X, Y, layer_dims, learning_rate=0.01, epochs=1000)

    # Test predictions
    predictions = predict(X, parameters)
    mse = np.mean((predictions - Y) ** 2)
    print(f"Final Mean Squared Error: {mse:.4f}")
