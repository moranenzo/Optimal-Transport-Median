# utils.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import ot

# 1. --- Data ---

def generate_data(n, distribution="uniform", l=1):
    """
    Generates random data according to the specified distribution in dimension 2.
    
    Args:
        n (int): Number of points to generate.
        distribution (str): The type of distribution (default is "uniform").
        l (float): Radius for the uniform distribution (only relevant for "uniform").
    
    Returns:
        np.array: Array of generated data.
    """
    d = 2  # Dimension (fixed here to 2)
    
    if distribution == "normal":
        # Parameters for the normal distribution (mean and standard deviation)
        mean = 0  # Default mean
        std = 1   # Default standard deviation
        return np.random.normal(loc=mean, scale=std, size=(n, d))
        
    elif distribution == "uniform":
        num_theta = 2*int(np.sqrt(n-1))
        theta = np.linspace (0 , 2 * np.pi , num_theta )
        rad = np.linspace(0.01, 1, int( (n-1)/num_theta )+1 )
        
        z = np . array ([[0 , 0]])
        for r in rad:
            for t in theta:
                x = r * np . cos (t)
                y = r * np . sin (t)
                z = np . vstack ((z , np . column_stack ((x , y)) ))
                
        return z[:n]

    elif distribution == "banana":
        # Generation of coordinates X and Phi
        X = -1 + 2 * np.random.rand(n)  # X between -1 and 1
        Phi = 2 * np.pi * np.random.rand(n)  # Phi between 0 and 2π
        
        # Calculation of radii R
        R = 0.2 * np.random.rand(n) * (1 + (1 - np.abs(X)) / 2)  # Radius adjusted according to X
        
        # Calculation of coordinates y according to the "banana" distribution
        z = np.column_stack((X + R * np.cos(Phi), X**2 + R * np.sin(Phi)))
        
        return z
    
    
    else:
        raise ValueError("Supported distributions: 'normal', 'uniform', 'banana'")





def process_ansur_data(file_path):
    """
    Transforms and cleans ANSUR data (Male or Female).
    
    Args:
        file_path (str): Path to the CSV file containing ANSUR data.
    
    Returns:
        np.array: Cleaned data array with columns for height (cm) and weight (kg).
    """
    # Read the ANSUR dataset
    ansur = pd.read_csv(file_path, sep=",", encoding="ISO-8859-1")
    
    # Select and rename relevant columns
    data = ansur[["Heightin", "weightkg"]].rename(columns={"Heightin": "Height (inches)", "weightkg": "Weight (kg)"})
    
    # Convert height from inches to centimeters
    data["Height (inches)"] *= 2.54
    data.rename(columns={"Height (inches)": "Height (cm)"}, inplace=True)
    
    # Convert weight from pounds to kilograms (divide by 10)
    data["Weight (kg)"] /= 10
    
    # Remove the last row (possible outlier)
    data = data.iloc[:-1]
    
    return np.array(data)




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


# 5. --- Wasserstein Distance ---

def wasserstein_distance(source, target, N):
    """Calculates the Wasserstein distance between two distributions."""
    a, b = np.ones(N), np.ones(N)
    M = ot.dist(source, target)
    return np.sqrt(ot.emd2(a, b, M))
