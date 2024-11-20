import numpy as np
import matplotlib.pyplot as plt
from utils import Z, dist

def main():
    N = 1000
    t_values = np.linspace(0, 10, 100)
    Z0 = Z(0, N)
    Z_values = [Z(t, N) for t in t_values]
    dist_values = np.array([dist(Z0, Zt, N) for Zt in Z_values])

    plt.figure(figsize=(8, 6))
    plt.plot(t_values, dist_values, label='Distance Wasserstein')
    plt.xlabel('t')
    plt.ylabel('Distance')
    plt.title('Distance Wasserstein entre Z(t) et Z(0)')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
