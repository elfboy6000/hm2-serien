import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Aufgabe 3: Polynominterpolation mit numpy.polyfit / polyval
# ============================================================

# Gegebene Datenpunkte: Anteil US-Haushalte mit Computer [%]
jahre = np.array([1981, 1984, 1989, 1993, 1997, 2000, 2001, 2003, 2004, 2010], dtype=float)
haushalte = np.array([0.5, 8.2, 15, 22.9, 36.6, 51, 56.3, 61.8, 65, 76.7])

n = len(jahre)        # 10 Datenpunkte
grad = n - 1          # Grad 9 -> Polynom geht exakt durch alle n Punkte

x_plot = np.arange(1975, 2020.1, 0.1)   # gemeinsame x-Achse für a) und b)

# ============================================================
# a) polyfit OHNE Mittelwertzentrierung
# ============================================================
koeff_a = np.polyfit(jahre, haushalte, grad)
y_plot_a = np.polyval(koeff_a, x_plot)

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(x_plot, y_plot_a, label='Interpolationspolynom (a)', color='steelblue')
ax1.plot(jahre, haushalte, 'ro', label='Datenpunkte', zorder=5)
ax1.set_xlim(1975, 2020)
ax1.set_ylim(-100, 250)
ax1.set_xlabel('Jahr')
ax1.set_ylabel('Haushalte mit Computer [%]')
ax1.set_title('Aufgabe 3a: Polynominterpolation (ohne Zentrierung)')
ax1.legend()
ax1.grid(True)
plt.tight_layout()

# Überprüfung: Abweichungen an den Datenpunkten
y_check_a = np.polyval(koeff_a, jahre)
print("a) Abweichungen des Polynoms von den Datenpunkten (ohne Zentrierung):")
for j, h, yc in zip(jahre, haushalte, y_check_a):
    print(f"   Jahr {int(j)}: Datenpunkt = {h:.1f} %, Polynom = {yc:.4f} %")

# ANTWORT a):
# Nein, das Polynom geht NICHT exakt durch alle Datenpunkte. Mit rohen
# Jahreszahlen (1981–2010) entstehen bei hohen Potenzen (x^9 ~ 2000^9)
# extreme numerische Rundungsfehler. Das Problem ist schlecht konditioniert
# (numpy gibt eine RankWarning über schlechte Konditionierung aus).
# Die berechneten Koeffizienten sind daher ungenau, und das Polynom weicht
# deutlich von den Datenpunkten ab.

# ============================================================
# b) polyfit MIT Mittelwertzentrierung (x - x.mean())
# ============================================================
jahre_mean = jahre.mean()
jahre_zentriert = jahre - jahre_mean

koeff_b = np.polyfit(jahre_zentriert, haushalte, grad)

# Für den Plot: x_plot ebenfalls zentrieren, dann polyval aufrufen
y_plot_b = np.polyval(koeff_b, x_plot - jahre_mean)

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(x_plot, y_plot_b, label='Interpolationspolynom (b, zentriert)', color='darkorange')
ax2.plot(jahre, haushalte, 'ro', label='Datenpunkte', zorder=5)
ax2.set_xlim(1975, 2020)
ax2.set_ylim(-100, 250)
ax2.set_xlabel('Jahr')
ax2.set_ylabel('Haushalte mit Computer [%]')
ax2.set_title('Aufgabe 3b: Polynominterpolation (mit Mittelwertzentrierung)')
ax2.legend()
ax2.grid(True)
plt.tight_layout()

# Überprüfung
y_check_b = np.polyval(koeff_b, jahre_zentriert)
print("\nb) Abweichungen des Polynoms von den Datenpunkten (mit Zentrierung):")
for j, h, yc in zip(jahre, haushalte, y_check_b):
    print(f"   Jahr {int(j)}: Datenpunkt = {h:.1f} %, Polynom = {yc:.6f} %")

# ANTWORT b):
# Mit Zentrierung (x - x.mean()) liegen die x-Werte im Bereich ca. -14 bis +10.
# Dadurch sind die hohen Potenzen (x^9) viel kleiner und das numerische
# Konditionierungsproblem wird beseitigt. Das Polynom geht nun praktisch exakt
# durch alle Datenpunkte (Abweichungen < 1e-10 %). Im Vergleich zu a) ist die
# Darstellung innerhalb des Intervalls korrekt; ausserhalb oszilliert das Polynom
# stark, aber das ist das erwartete Verhalten eines Polynoms hoher Ordnung.

# ============================================================
# c) Schätzwert für 2020
# ============================================================
y_2020 = np.polyval(koeff_b, 2020 - jahre_mean)
print(f"\nc) Schätzwert für Jahr 2020: {y_2020:.2f} %")

# ANTWORT c):
# Der Schätzwert für 2020 ist physikalisch völlig unrealistisch (weit ausserhalb
# des sinnvollen Bereichs 0–100%). Polynome hoher Ordnung zeigen ausserhalb des
# Interpolationsintervalls ein explosives Schwingungsverhalten (Runge-Phänomen)
# und divergieren sehr schnell. Sie sind daher für Extrapolation (Schätzungen
# ausserhalb des Bereichs der vorhandenen Daten) NICHT geeignet.

# ============================================================
# d) Lagrange-Interpolation (Funktion aus Aufgabe 2, erweitert auf Vektoren)
# ============================================================

def lagrange_int(x, y, x_int):
    """
    Lagrange-Interpolation. x_int kann ein Skalar oder ein Vektor sein.

    Args:
        x (list or np.array): x-Koordinaten der Datenpunkte.
        y (list or np.array): y-Koordinaten der Datenpunkte (darf NaN enthalten).
        x_int (float or np.array): x-Koordinate(n) zum Interpolieren.

    Returns:
        float or np.array: Interpolierte y-Werte.
    """
    x_points = np.array(x, dtype=float)
    y_points = np.array(y, dtype=float)

    # Entferne NaN-Werte aus den Datenpunkten
    valid = ~np.isnan(y_points)
    x_known = x_points[valid]
    y_known = y_points[valid]

    # x_int als Array behandeln (ermöglicht Vektoreingabe)
    scalar_input = np.isscalar(x_int)
    x_int = np.atleast_1d(np.array(x_int, dtype=float))
    y_int = np.zeros_like(x_int)

    m = len(x_known)
    for j in range(m):
        # Lagrange-Basispolynom L_j(x_int) – vektorisiert
        p = np.ones_like(x_int)
        for i in range(m):
            if i != j:
                p *= (x_int - x_known[i]) / (x_known[j] - x_known[i])
        y_int += y_known[j] * p

    return float(y_int[0]) if scalar_input else y_int


# Lagrange-Werte für x ∈ [1981, 2010]
x_lagrange = np.arange(1981, 2010.1, 0.1)
y_lagrange = lagrange_int(jahre, haushalte, x_lagrange)

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(x_lagrange, y_lagrange,
         label='Lagrange-Interpolation [1981, 2010]', color='blue', linewidth=2)
ax3.plot(x_plot, y_plot_b,
         label='polyfit – zentriert (b) [1975, 2020]', color='darkorange',
         linestyle='--', linewidth=1.5)
ax3.plot(jahre, haushalte, 'ro', label='Datenpunkte', zorder=5)
ax3.set_xlim(1975, 2020)
ax3.set_ylim(-100, 250)
ax3.set_xlabel('Jahr')
ax3.set_ylabel('Haushalte mit Computer [%]')
ax3.set_title('Aufgabe 3d: Lagrange vs. polyfit (b)')
ax3.legend()
ax3.grid(True)
plt.tight_layout()

# ANTWORT d):
# Innerhalb des gemeinsamen Intervalls [1981, 2010] sind Lagrange-Interpolation
# und polyfit (mit Zentrierung) praktisch identisch – beide berechnen dasselbe
# eindeutige Interpolationspolynom 9. Grades durch die 10 Datenpunkte.
# Der wesentliche Unterschied: polyfit (b) zeigt auch das Verhalten ausserhalb
# [1981, 2010], wo das Polynom stark oszilliert und divergiert. Lagrange wurde
# hier nur im Interpolationsintervall berechnet, daher kein Überschwingen sichtbar.
# Fazit: Beide Methoden sind mathematisch äquivalent; der Unterschied liegt nur
# in der numerischen Implementierung und der gewählten Darstellung.

plt.show()
