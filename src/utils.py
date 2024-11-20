import numpy as np
import ot

def dist(source, cible, N):
    """Calcule la distance Wasserstein entre deux distributions."""
    a, b = np.ones(N), np.ones(N)
    M = ot.dist(source, cible)
    return np.sqrt(ot.emd2(a, b, M))

def Z(t, N):
    """Génère une distribution gaussienne multivariée centrée."""
    mean = [t, t]
    cov = [[1, 0], [0, 1]]
    return np.random.multivariate_normal(mean, cov, N)
