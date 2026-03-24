import numpy as np
import matplotlib.pyplot as plt


def Yanik_Michael_S5_Aufg2(x, y, xx, plot=True):
    """
    Berechnet und plottet die natürliche kubische Splinefunktion S(x).

    Parameter
    ----------
    x  : array-like, (n+1) Stützstellen (aufsteigend sortiert)
    y  : array-like, (n+1) Stützwerte
    xx : array-like, Auswertungsstellen innerhalb [x[0], x[n]]

    Rückgabe
    --------
    yy : ndarray, S(xx)
    """
    x  = np.asarray(x, dtype=float)
    y  = np.asarray(y, dtype=float)
    xx = np.asarray(xx, dtype=float)
    n  = len(x) - 1          # Anzahl Intervalle

    # ------------------------------------------------------------------
    # Schritt 1 & 2: a-Koeffizienten und Intervallbreiten
    # ------------------------------------------------------------------
    a = y.copy()             # ai = yi
    h = np.diff(x)           # hi = x[i+1] - x[i]

    # ------------------------------------------------------------------
    # Schritt 3 & 4: c-Koeffizienten aus tridiagonalem LGS
    #   c[0] = 0,  c[n] = 0  (natürliche Randbedingung)
    # ------------------------------------------------------------------
    # Aufbau der (n-1) x (n-1) Matrix A und rechten Seite z
    size = n - 1
    A = np.zeros((size, size))
    z = np.zeros(size)

    for i in range(size):
        # Hauptdiagonale: 2*(h[i] + h[i+1])
        A[i, i] = 2.0 * (h[i] + h[i + 1])
        # rechte Seite
        z[i] = 3.0 * (a[i + 2] - a[i + 1]) / h[i + 1] \
             - 3.0 * (a[i + 1] - a[i])     / h[i]
        # Nebendiagonalen
        if i > 0:
            A[i, i - 1] = h[i]
        if i < size - 1:
            A[i, i + 1] = h[i + 1]

    # Lösung mit Gauss-Algorithmus (numpy)
    c_inner = np.linalg.solve(A, z)

    # Vollständiger c-Vektor (inkl. Randbedingungen)
    c = np.zeros(n + 1)
    c[1:n] = c_inner          # c[0] = c[n] = 0 bereits gesetzt

    # ------------------------------------------------------------------
    # Schritt 5 & 6: b- und d-Koeffizienten
    # ------------------------------------------------------------------
    b = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        b[i] = (a[i + 1] - a[i]) / h[i] - h[i] / 3.0 * (c[i + 1] + 2.0 * c[i])
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])

    # ------------------------------------------------------------------
    # Auswertung S(xx): finde für jedes xx das zugehörige Intervall
    # ------------------------------------------------------------------
    yy = np.zeros_like(xx)
    for k, xk in enumerate(xx):
        # Intervall-Index: letztes i mit x[i] <= xk
        i = np.searchsorted(x, xk, side='right') - 1
        i = int(np.clip(i, 0, n - 1))
        dx = xk - x[i]
        yy[k] = a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3

    # ------------------------------------------------------------------
    # Plot (nur wenn explizit gewünscht)
    # ------------------------------------------------------------------
    if not plot:
        return yy

    plt.figure(figsize=(8, 5))
    plt.plot(xx, yy, 'b-', label='Natürlicher kub. Spline')
    plt.plot(x,  y,  'ro', markersize=8, label='Stützpunkte')
    plt.xlabel('x')
    plt.ylabel('S(x)')
    plt.title('Natürliche kubische Splinefunktion')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return yy


# ======================================================================
# Überprüfung mit den Stützpunkten aus Aufgabe 1
# ======================================================================
if __name__ == '__main__':
    x_test  = np.array([4, 6, 8, 10], dtype=float)
    y_test  = np.array([6, 3, 9,  0], dtype=float)
    xx_test = np.linspace(x_test[0], x_test[-1], 300)

    yy_test = Yanik_Michael_S5_Aufg2(x_test, y_test, xx_test)
    print("S(5)  =", Yanik_Michael_S5_Aufg2(x_test, y_test, np.array([5.0]),  plot=False)[0])
    print("S(9)  =", Yanik_Michael_S5_Aufg2(x_test, y_test, np.array([9.0]),  plot=False)[0])