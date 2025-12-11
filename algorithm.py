'''
This module implements linear regression from scratch using batch gradient 
descent. The algorithm starts with θ₀ = θ₁ = 0 and repeatedly:
1. Computes predictions using current parameters
2. Calculates cost (how far predictions are from actual values)
3. Computes gradients (direction to move parameters to reduce cost)
4. Updates parameters by taking a small step (learning rate) in negative 
   gradient direction

This process continues until the cost converges to a minimum, yielding
optimal parameters that best fit the training data.
'''
import numpy as np

class linearRegressionGD:
    def __init__(self, learning_rate=0.02, n_iterations=2000):
        self.learning_rate = learning_rate  # Step size for parameter updates
        self.n_iterations = n_iterations    # Max number of iterations

        # Parameters we're trying to learn
        self.theta_0 = 0.0 # Intercept
        self.theta_1 = 0.0 # Slope

        # Track progress
        self.cost_history = []

    def predict(self, X):
        """Compute predictions: h(x) = θ₀ + θ₁·x"""
        return self.theta_0 + self.theta_1 * X
    
    def fit(self, X, y):
        """Train using gradient descent"""
        m = len(X)

        # Repeats until convergence
        for i in range(self.n_iterations):
            # Predictions and errors
            predictions = self.predict(X)
            errors = predictions - y

            # Compute cost using squared error: J(θ) = (1/2m) Σ(error²)
            cost = (1 / (2 * m)) * np.sum(errors ** 2)
            self.cost_history.append(cost)

            # Compute gradients: calculate the partial derivatives
            grad_0 = (1 / m) * np.sum(errors) # average error across all examples
            grad_1 = (1 / m) * np.sum(errors * X) # error weighted by input values

            # Update parameters
            self.theta_0 -= self.learning_rate * grad_0
            self.theta_1 -= self.learning_rate * grad_1

            # Print progress
            if i % 200 == 0:
                print(f"Iteration {i}: Cost={cost:.4f}, θ₀={self.theta_0:.4f}, θ₁={self.theta_1:.4f}")
        
        print(f"\nFinal: θ₀ = {self.theta_0:.4f}, θ₁ = {self.theta_1:.4f}")