import numpy as np
from numpy.linalg import norm

def f(x):
    x1, x2, x3 = x
    return np.array([
        x1 + x2**2 - x3**2 - 13,
        np.log(x2 / 4.0) + np.exp(0.5 * x3 - 1.0) - 1.0,
        (x2 - 3.0)**2 - x3**3 + 7.0
    ], dtype=float)

def J(x):
    x1, x2, x3 = x
    return np.array([
        [1.0, 2.0 * x2,      -2.0 * x3],
        [0.0, 1.0 / x2,       0.5 * np.exp(0.5 * x3 - 1.0)],
        [0.0, 2.0 * (x2-3.0), -3.0 * x3**2]
    ], dtype=float)

def damped_newton(x0, tol=1e-5, max_iter=50, c=1e-4):
    xk = x0.astype(float).copy()

    for k in range(max_iter):
        Fx = f(xk)
        if norm(Fx, 2) < tol:
            return xk, k

        # Newton-Richtung
        dx = np.linalg.solve(J(xk), -Fx)

        # Dämpfung / Backtracking line search
        lam = 1.0
        Fx_norm = norm(Fx, 2)

        while True:
            x_new = xk + lam * dx

            # Domain-Check: log(x2/4) braucht x2 > 0
            if x_new[1] <= 0:
                lam *= 0.5
            else:
                if norm(f(x_new), 2) <= (1 - c * lam) * Fx_norm:
                    break
                lam *= 0.5

            if lam < 1e-8:
                # gives up so that it does not remain halved forever
                break

        xk = x_new

    return xk, max_iter

x0 = np.array([1.5, 3.0, 2.5])
sol, iterations = damped_newton(x0)

print(f"Lösung: x{iterations} = {sol}")