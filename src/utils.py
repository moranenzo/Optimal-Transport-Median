# utils.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import ot

# 1. --- Simulation de données ---

def ech_X(n, d, loi="normale", params=None):
    """
    Génère des données aléatoires selon la loi spécifiée.
    
    Args:
        n (int): Nombre de points à générer.
        d (int): Dimension des points.
        loi (str): La loi de distribution (par défaut "normale").
        params (dict): Paramètres supplémentaires pour la loi (par exemple, moyenne, covariance).
    
    Returns:
        np.array: Tableau des données générées.
    """
    if loi == "normale":
        return np.random.normal(loc=params.get('mean', 0), scale=params.get('std', 1), size=(n, d))
    elif loi == "uniforme":
        return np.random.uniform(low=params.get('low', 0), high=params.get('high', 1), size=(n, d))
    elif loi == "banane":
        return np.column_stack([np.sin(np.linspace(0, 2*np.pi, n)), np.linspace(0, 1, n)])
    else:
        raise ValueError("Lois supportées : 'normale', 'uniforme', 'banane'")


# 2. --- Calcul du transport optimal ---

def transport_optimal(X, Y, dist_type='euclidean'):
    """
    Calcule le transport optimal entre deux ensembles de points X et Y en utilisant l'algorithme de transport optimal de type transport de masse.
    
    Args:
        X (np.array): Ensemble de points source (n, d).
        Y (np.array): Ensemble de points cible (m, d).
        dist_type (str): Type de distance ('euclidean' par défaut).
        
    Returns:
        tuple: Le coût optimal, et le transport (matrice de correspondances).
    """
    dist_matrix = cdist(X, Y, metric=dist_type)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    cost = dist_matrix[row_ind, col_ind].sum()
    
    # Créer une matrice de transport avec des poids pour chaque correspondance
    transport_matrix = np.zeros_like(dist_matrix)
    transport_matrix[row_ind, col_ind] = 1
    
    return cost, transport_matrix


def transport_optimal_median(X, Y, dist_type='euclidean'):
    """
    Calcule la médiane des points par transport optimal entre deux ensembles X et Y.
    
    Args:
        X (np.array): Ensemble de points source (n, d).
        Y (np.array): Ensemble de points cible (m, d).
        dist_type (str): Type de distance ('euclidean' par défaut).
    
    Returns:
        np.array: La médiane obtenue par transport optimal.
    """
    _, transport_matrix = transport_optimal(X, Y, dist_type)
    median = transport_matrix.mean(axis=0)
    return median


# 3. --- Visualisation des résultats ---

def plot_transport(X, Y, transport_matrix, title="Transport Optimal"):
    """
    Visualise le transport optimal entre deux ensembles de points X et Y.
    
    Args:
        X (np.array): Ensemble de points source (n, d).
        Y (np.array): Ensemble de points cible (m, d).
        transport_matrix (np.array): Matrice de transport de correspondance.
        title (str): Titre du graphique.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], label="Points Source", color='blue')
    plt.scatter(Y[:, 0], Y[:, 1], label="Points Cible", color='red')
    
    for i in range(len(X)):
        for j in range(len(Y)):
            if transport_matrix[i, j] > 0:
                plt.plot([X[i, 0], Y[j, 0]], [X[i, 1], Y[j, 1]], 'k-', alpha=0.5)
    
    plt.title(title)
    plt.legend()
    plt.show()


def plot_median(X, Y, median, title="Médiane par Transport Optimal"):
    """
    Affiche la médiane obtenue par transport optimal sur les points.
    
    Args:
        X (np.array): Ensemble de points source (n, d).
        Y (np.array): Ensemble de points cible (m, d).
        median (np.array): La médiane obtenue.
        title (str): Titre du graphique.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], label="Points Source", color='blue')
    plt.scatter(Y[:, 0], Y[:, 1], label="Points Cible", color='red')
    plt.scatter(median[0], median[1], color='green', label='Médiane', marker='x')
    
    plt.title(title)
    plt.legend()
    plt.show()


# 4. --- Calcul des quantiles ---

def quantiles(data, q):
    """
    Calcule les quantiles d'un ensemble de données.
    
    Args:
        data (np.array): Données d'entrée.
        q (list): Liste des quantiles à calculer (par exemple, [0.25, 0.5, 0.75]).
    
    Returns:
        np.array: Les quantiles demandés.
    """
    return np.percentile(data, [x * 100 for x in q], axis=0)


# 5. --- Autres utilitaires ---

def boule(n, d, rayon=1):
    """
    Génère une boule de dimension d et de rayon spécifié.
    
    Args:
        n (int): Nombre de points à générer.
        d (int): Dimension de la boule.
        rayon (float): Rayon de la boule.
    
    Returns:
        np.array: Tableau des points générés dans la boule.
    """
    points = np.random.randn(n, d)
    points = points / np.linalg.norm(points, axis=1)[:, np.newaxis]  # Normaliser pour être dans la boule unitaire
    points = points * np.random.uniform(0, rayon, n)[:, np.newaxis]  # Appliquer un rayon
    return points


# 6. --- Distance Wasserstein ---

def dist(source, cible, N):
    """Calcule la distance Wasserstein entre deux distributions."""
    a, b = np.ones(N), np.ones(N)
    M = ot.dist(source, cible)
    return np.sqrt(ot.emd2(a, b, M))


# 7. --- Génération de données gaussiennes ---

def Z(t, N):
    """Génère une distribution gaussienne multivariée centrée."""
    mean = [t, t]
    cov = [[1, 0], [0, 1]]
    return np.random.multivariate_normal(mean, cov, N)
