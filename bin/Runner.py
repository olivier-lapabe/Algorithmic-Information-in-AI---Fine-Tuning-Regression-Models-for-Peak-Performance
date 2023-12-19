
from src.Model.model_tuning import ModelRunner, plot_data
import numpy as np

import numpy as np


def generate_synthetic_data(num_samples=1000, noise_level=1.0, coeffs=[1, 2, 3, -4], x_range=(0, 10),
                            random_state=None):
    """
    Generates synthetic data for polynomial regression.

    Parameters:
    - num_samples (int): Number of data points to generate.
    - noise_level (float): Standard deviation of Gaussian noise added to the data.
    - coeffs (list): Coefficients for the polynomial. For example, [1, 2, 3, -4] represents 1 + 2x + 3x^2 - 4x^3.
    - x_range (tuple): The range (min, max) of x values.
    - random_state (int or None): Seed for the random number generator.

    Returns:
    - x (ndarray): Feature variable.
    - y (ndarray): Target variable.
    """

    if random_state is not None:
        np.random.seed(random_state)

    # Generate feature variable x
    x = np.random.rand(num_samples, 1) * (x_range[1] - x_range[0]) + x_range[0]

    # Calculate target variable y based on the polynomial and add noise
    y = np.zeros((num_samples, 1))
    for i, coeff in enumerate(coeffs):
        y += coeff * x ** i
    y += np.random.randn(num_samples, 1) * noise_level

    return x, y


def main():
    # Example custom dataset
    # Replace this with code to load your dataset
    X = np.random.rand(1000, 1)
    y = 2 * X.squeeze() + 1 + np.random.randn(1000) * 0.1

    # Plot the custom dataset
    plot_data(X, y)

    # Create and run the model runner with the custom dataset
    runner = ModelRunner(X, y, train_size=0.8, max_degree=5)
    runner.run()

    # Plot the complexities and MSEs
    # You can call the plot_complexities function using runner's attributes

if __name__ == main():
    main()


# -----------------------------------------------------------------------------
# generate_synthetic_data
# -----------------------------------------------------------------------------
def generate_synthetic_data(d):
    if d == 1:
        np.random.seed(0)
        x = np.random.rand(1000, 1) * 10 - 5
        noise = np.random.randn(1000, 1) * 100
        y = 1 + 2 * x + 3 * x ** 2 - 4 * x ** 3 + noise
        return x, y

    if d == 2:
        np.random.seed(0)
        x = np.random.rand(1000, 1) - 0.5
        noise = np.random.randn(1000, 1)
        y = 1 - 1 * np.exp(x) + 2 * np.exp(x) ** 2 - 4 * np.exp(x) ** 3 + noise
        return x, y

    if d == 3:
        np.random.seed(0)
        x = np.random.rand(1000, 1) - 0.5
        noise = np.random.randn(1000, 1) * 10
        y = 1 - 3 * x ** 2 - 3 * np.exp(x) ** 2 + noise
        return x, y

    if d == 4:
        np.random.seed(0)
        x = np.random.rand(1000, 1) * 10 - 5
        noise = np.random.randn(1000, 1) * 10
        y = 1 - 4 * x + 1 * x ** 2 + noise
        return x, y

    if d == 5:
        np.random.seed(0)
        x = np.random.rand(1000, 1) * 10 - 5
        noise = np.random.randn(1000, 1) * 100
        y = 1 - 20 * x + 0.3 * x ** 2 + 0.4 * x ** 3 + 1 * x ** 4 + noise
        return x, y

    if d == 6:
        np.random.seed(0)
        x = np.random.rand(1000, 1)
        noise = np.random.randn(1000, 1) * 10
        y = 1 - 1 * np.log(x) + 2 * np.log(x) ** 2 + noise
        return x, y