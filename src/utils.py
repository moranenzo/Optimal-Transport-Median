# utils.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import ot

# 1. --- Data Simulation ---

def generate_data(n, d, distribution="normal", params=None):
    """
    Generates random data according to the specified distribution.
    
    Args:
        n (int): Number of points to generate.
        d (int): Dimension of the points.
        distribution (str): The type of distribution (default is "normal").
        params (dict): Additional parameters for the distribution (e.g., mean, covariance).
    
    Returns:
        np.array: Array of generated data.
    """
    if distribution == "normal":
        return np.random.normal(loc=params.get('mean', 0), scale=params.get('std', 1), size=(n, d))
    elif distribution == "uniform":
        return np.random.uniform(low=params.get('low', 0), high=params.get('high', 1), size=(n, d))
    elif distribution == "banana":
        return np.column_stack([np.sin(np.linspace(0, 2*np.pi, n)), np.linspace(0, 1, n)])
    else:
        raise ValueError("Supported distributions: 'normal', 'uniform', 'banana'")


# 2. --- Optimal Transport Calculation ---

def optimal_transport(X, Y, dist_type='euclidean'):
    """
    Computes the optimal transport between two sets of points X and Y using the mass transport algorithm.
    
    Args:
        X (np.array): Source set of points (n, d).
        Y (np.array): Target set of points (m, d).
        dist_type (str): Distance type ('euclidean' by default).
        
    Returns:
        tuple: The optimal cost and the transport matrix (matching matrix).
    """
    dist_matrix = cdist(X, Y, metric=dist_type)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    cost = dist_matrix[row_ind, col_ind].sum()
    
    # Create a transport matrix with weights for each match
    transport_matrix = np.zeros_like(dist_matrix)
    transport_matrix[row_ind, col_ind] = 1
    
    return cost, transport_matrix


def optimal_transport_median(X, Y, dist_type='euclidean'):
    """
    Computes the median of points via optimal transport between two sets X and Y.
    
    Args:
        X (np.array): Source set of points (n, d).
        Y (np.array): Target set of points (m, d).
        dist_type (str): Distance type ('euclidean' by default).
    
    Returns:
        np.array: The median obtained via optimal transport.
    """
    _, transport_matrix = optimal_transport(X, Y, dist_type)
    median = transport_matrix.mean(axis=0)
    return median


# 3. --- Result Visualization ---

def plot_transport(X, Y, transport_matrix, title="Optimal Transport"):
    """
    Visualizes the optimal transport between two sets of points X and Y.
    
    Args:
        X (np.array): Source set of points (n, d).
        Y (np.array): Target set of points (m, d).
        transport_matrix (np.array): Matching transport matrix.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], label="Source Points", color='blue')
    plt.scatter(Y[:, 0], Y[:, 1], label="Target Points", color='red')
    
    for i in range(len(X)):
        for j in range(len(Y)):
            if transport_matrix[i, j] > 0:
                plt.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]], 'k-', alpha=0.5)
    
    plt.title(title)
    plt.legend()
    plt.show()


def plot_median(X, Y, median, title="Median by Optimal Transport"):
    """
    Displays the median obtained by optimal transport on the points.
    
    Args:
        X (np.array): Source set of points (n, d).
        Y (np.array): Target set of points (m, d).
        median (np.array): The median obtained.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], label="Source Points", color='blue')
    plt.scatter(Y[:, 0], Y[:, 1], label="Target Points", color='red')
    plt.scatter(median[0], median[1], color='green', label='Median', marker='x')
    
    plt.title(title)
    plt.legend()
    plt.show()


# 4. --- Quantile Calculation ---

def quantiles(data, q):
    """
    Calculates the quantiles of a dataset.
    
    Args:
        data (np.array): Input data.
        q (list): List of quantiles to calculate (e.g., [0.25, 0.5, 0.75]).
    
    Returns:
        np.array: The requested quantiles.
    """
    return np.percentile(data, [x * 100 for x in q], axis=0)


# 5. --- Other Utilities ---

def ball(n, d, radius=1):
    """
    Generates points inside a ball of dimension d and a specified radius.
    
    Args:
        n (int): Number of points to generate.
        d (int): Dimension of the ball.
        radius (float): Radius of the ball.
    
    Returns:
        np.array: Array of points inside the ball.
    """
    points = np.random.randn(n, d)
    points = points / np.linalg.norm(points, axis=1)[:, np.newaxis]  # Normalize to be within the unit ball
    points = points * np.random.uniform(0, radius, n)[:, np.newaxis]  # Apply the radius
    return points


# 6. --- Wasserstein Distance ---

def wasserstein_distance(source, target, N):
    """Calculates the Wasserstein distance between two distributions."""
    a, b = np.ones(N), np.ones(N)
    M = ot.dist(source, target)
    return np.sqrt(ot.emd2(a, b, M))


# 7. --- Gaussian Data Generation ---

def Z(t, N):
    """Generates a centered multivariate Gaussian distribution."""
    mean = [t, t]
    cov = [[1, 0], [0, 1]]
    return np.random.multivariate_normal(mean, cov, N)
