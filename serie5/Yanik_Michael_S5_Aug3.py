import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# Eigene Funktion aus Aufgabe 2 importieren
from Yanik_Michael_S5_Aufg2 import Yanik_Michael_S5_Aufg2

# ======================================================================
# Daten
# ======================================================================
t = np.array([1900, 1910, 1920, 1930, 1940, 1950,
              1960, 1970, 1980, 1990, 2000, 2010], dtype=float)
p = np.array([75.995, 91.972, 105.711, 123.203, 131.669, 150.697,
              179.323, 203.212, 226.505, 249.633, 281.422, 308.745])

tt = np.linspace(t[0], t[-1], 500)   # feines Gitter für Plots

# ======================================================================
# a) Eigene Spline-Funktion (Aufgabe 2)
# ======================================================================
yy_own = Yanik_Michael_S5_Aufg2(t, p, tt)     # plottet intern ebenfalls

# ======================================================================
# b) scipy CubicSpline (natürliche Randbedingung: bc_type='natural')
# ======================================================================
cs = interpolate.CubicSpline(t, p, bc_type='natural')
yy_scipy = cs(tt)

# ======================================================================
# c) Polynom 11. Grades via numpy.polyfit
#    Zeitreihe zuerst auf [0, 110] verschieben, damit die Vandermonde-
#    Matrix besser konditioniert ist (große Zahlen wie 1900^11 würden
#    zu extremen Konditionszahlen führen).
# ======================================================================
t_shifted = t - t[0]              # 0, 10, 20, ..., 110
tt_shifted = tt - t[0]

coeffs = np.polyfit(t_shifted, p, deg=11)
yy_poly = np.polyval(coeffs, tt_shifted)

# ======================================================================
# Vergleichsplot aller drei Methoden
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Linke Achse: alle drei Kurven überlagert ---
ax = axes[0]
ax.plot(tt, yy_own,   'b-',  lw=2,   label='Eigener Spline (Aufg. 2)')
ax.plot(tt, yy_scipy, 'g--', lw=2,   label='scipy CubicSpline (natural)')
ax.plot(tt, yy_poly,  'r:',  lw=2,   label='numpy.polyfit  Grad 11')
ax.plot(t,  p,        'ko',  ms=7,   label='Stützpunkte')
ax.set_xlabel('Jahr')
ax.set_ylabel('Bevölkerung (Mio.)')
ax.set_title('Bevölkerung USA – Vergleich der Interpolationsmethoden')
ax.legend(fontsize=9)
ax.grid(True)

# --- Rechte Achse: Differenz zwischen eigenem und scipy-Spline ---
ax2 = axes[1]
ax2.plot(tt, np.abs(yy_own - yy_scipy), 'b-', lw=1.5)
ax2.set_xlabel('Jahr')
ax2.set_ylabel('|Eigener – scipy| (Mio.)')
ax2.set_title('Abweichung: eigener Spline vs. scipy')
ax2.grid(True)

plt.tight_layout()
plt.show()

# ======================================================================
# Kurzausgabe: Werte bei ausgewählten Jahren
# ======================================================================
years_check = [1905, 1935, 1955, 1975, 2005]
print(f"{'Jahr':>6}  {'Eigener':>10}  {'scipy':>10}  {'Polynom':>10}")
print('-' * 42)
for yr in years_check:
    idx   = np.argmin(np.abs(tt - yr))
    print(f"{yr:>6}  {yy_own[idx]:>10.3f}  {yy_scipy[idx]:>10.3f}  {yy_poly[idx]:>10.3f}")