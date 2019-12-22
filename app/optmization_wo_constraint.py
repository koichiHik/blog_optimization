
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

from enum import IntEnum

def compute_original_func_values(x, y):
    Z = -X**3 - 3*X*Y**2 + Y**3 + 3*X
    return Z

def compute_gradient_vectors(x, y):

    x1 = -3*x**2 - 3*y**2 + 3
    x2 = -6*x*y + 3*y**2
    return np.array([[x1], [x2]])

def compute_hessian(x, y):

    h = np.zeros((2, 2))
    h[0, 0] = -6*x
    h[0, 1] = -6*y   
    h[1, 0] = -6*y
    h[1, 1] = -6*x + 6*y
    
    return h    

def find_extremum(x, y):

    # Derivative Values.
    X = np.array([[x], [y]])

    # Store Trajectory of X.
    X_traj = X

    for _ in range(1000):

        # Compute Jacobian (Jacobian of gradient vector is Hessian of original function.)
        jaco = compute_hessian(X[0, 0], X[1, 0])

        # Compute Gradient Vector
        grad_v = compute_gradient_vectors(X[0, 0], X[1, 0])

        # Parameter Correction Vector
        dX = np.matmul(-np.linalg.inv(jaco), grad_v)
        res = np.linalg.norm(dX)

        # Parameter Correction.
        X = X + dX
        X_traj = np.append(X_traj, X, axis=1)
        if (res < 0.000001):
            break

    return X_traj

class Definiteness(IntEnum):

    PositiveDefinite = 0
    NegativeDefinite = 1
    Indefinite = 2

def definiteness_check(x, y):

    h = compute_hessian(x, y)
    eigs, _ = np.linalg.eig(h)

    if (np.where(eigs > 0)[0].shape[0] == eigs.shape[0]):
        return Definiteness.PositiveDefinite
    if (np.where(eigs < 0)[0].shape[0] == eigs.shape[0]):
        return Definiteness.NegativeDefinite

    return Definiteness.Indefinite

if __name__ == "__main__":


    # 1. Find Local Minimum
    x = 0.5
    y = 0.0
    X1 = find_extremum(x, y)
    print(X1)

    # 2. Find Local Maximum
    x = -0.5
    y = -0.0
    X2 = find_extremum(x, y)

    # 3. Definiteness Check
    X1_color = 'r' if definiteness_check(X1[0, X1.shape[1] - 1], X1[1, X1.shape[1] - 1]) == Definiteness.NegativeDefinite else 'b'
    X2_color = 'r' if definiteness_check(X2[0, X2.shape[1] - 1], X2[1, X2.shape[1] - 1]) == Definiteness.NegativeDefinite else 'b'

    # 4. Plot Curvature
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212)
    x = y = np.arange(-2.0, 2.0, 0.05)
    X, Y = np.meshgrid(x, y)
    Z = compute_original_func_values(X, Y)

    Z[np.logical_or(Z < -3, 3 < Z)] = np.nan
    ax1.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
    ax2.contour(X, Y, Z)
    ax2.scatter(X1[0, 0:X1.shape[1]-1], X1[1, 0:X1.shape[1]-1], c=X1_color, s=10)
    ax2.scatter(X1[0, -1], X1[1, -1], c=X1_color, s=100)
    ax2.scatter(X2[0, 0:X2.shape[1]-1], X2[1, 0:X2.shape[1]-1], c=X2_color, s=10)
    ax2.scatter(X2[0, -1], X2[1, -1], c=X2_color, s=100)

    plt.show()
