'''
This module will generate synthetic linear data with controlled noise.
I will be using the parameters 2.5 for the slope and 5.0 as the 
y-intercept. The data will follow the relationship y = 2.5x + 5.0 + noise.
We add noise to make the problem realistic since real data is never perfectly
linear.
'''
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples = 150, true_slope = 2.5, true_intercept = 5.0, noise_std = 2.0, seed = 42):
    '''Generate sample data with a linear trend'''
    # Set random seed so we get the same data everytime we run
    np.random.seed(seed)

    # Generate X values evenly spaced between 0 and 10
    X = np.linspace(0, 10, n_samples)

    # Calculate the "perfect" y values (without noise)
    # This is our true linear relationship
    y_perfect = true_slope * X + true_intercept

    # Add Gaussian (normal) noise to make the data realistic
    # np.random.randn generates standard normal (mean = 0, std = 1)
    # We multiply by noise_std to get our desired standard deviation
    noise = np.random.randn(n_samples) * noise_std

    # Final y values = perfect line + noise
    y = y_perfect + noise

    # Store the true parameters so we can compare later
    true_params = {
        'slope': true_slope,
        'intercept': true_intercept,
        'noise_std': noise_std
    }
    return X, y, true_params

def visualize_data(X, y, true_params, save_path='data_plot.png'):
    """Create a scatter plot of the generated data with the true line."""
    plt.figure(figsize=(10, 6))
    
    # Plot the noisy data points
    plt.scatter(X, y, alpha=0.6, s=30, label='Generated data (with noise)', color='blue')
    
    # Plot the true underlying line (without noise)
    y_true = true_params['slope'] * X + true_params['intercept']
    plt.plot(X, y_true, 'r--', linewidth=2, 
             label=f"True line: y = {true_params['slope']}x + {true_params['intercept']}")
    
    plt.xlabel('X (feature)', fontsize=12)
    plt.ylabel('y (target)', fontsize=12)
    plt.title('Synthetic Linear Dataset with Gaussian Noise', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved as '{save_path}'")
    
    plt.close()  # Close to free memory

# Test the function
if __name__ == "__main__":
    # Generate data
    X, y, true_params = generate_data(
        n_samples=150,
        true_slope=2.5,
        true_intercept=5.0,
        noise_std=2.0,
        seed=42
    )
    
    # Create and save visualization
    visualize_data(X, y, true_params)
