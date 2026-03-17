import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

t = np.linspace(0, 10, 100)
x = np.cos(t)

def w(x, t, c=1):
    return np.sin(x + c * t)

def v(x, t, c=1):
    return np.sin(x + c * t) + np.cos(2 * x + 2 * c * t)

if __name__ == "__main__":
    # grid for x and t
    x = np.linspace(-5, 5, 120)
    t = np.linspace(0, 10, 120)
    X, T = np.meshgrid(x, t)

    W = w(X, T, c=1)
    V = v(X, T, c=1)

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_wireframe(X, T, W, rstride=3, cstride=3, color='tab:blue')
    ax1.set_title('w(x,t) = sin(x + c t)')
    ax1.set_xlabel('x'); ax1.set_ylabel('t'); ax1.set_zlabel('w')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_wireframe(X, T, V, rstride=3, cstride=3, color='tab:orange')
    ax2.set_title('v(x,t) = sin(x + c t) + cos(2x + 2 c t)')
    ax2.set_xlabel('x'); ax2.set_ylabel('t'); ax2.set_zlabel('v')

    plt.tight_layout()
    plt.show()
