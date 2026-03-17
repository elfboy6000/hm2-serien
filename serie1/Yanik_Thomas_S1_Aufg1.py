import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Aufgabe 1 a)
# Konstanten v0 und g fallen weg, Wurfweite hängt von sin(2*alpha) ab. 
# sin(2*alpha) <= 1, also alpha = pi/4(45°).
# Sinnvoller Definitionsbereich: alpha in [0, pi/2].

# Funktion W(v0, alpha) = v0^2 * sin(2*alpha) / g
def W(v0, alpha):
    return (v0**2 * np.sin(2 * alpha)) / 9.81  # g = 9.81 m/s^2

# Definitionsbereiche
v0 = np.linspace(0, 100, 100)  # v0 in [0, 100] m/s
alpha = np.linspace(0, np.pi/2, 100)  # alpha in [0, pi/2] rad (0° bis 90°)

# Gitter
[V0, Alpha] = np.meshgrid(v0, alpha)
W_values = W(V0, Alpha)

# 1. Plot: Wireframe
fig1 = plt.figure(1, figsize=(10, 7))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_wireframe(V0, Alpha, W_values, rstride=5, cstride=5)
ax1.set_xlabel('Anfangsgeschwindigkeit v0 [m/s]')
ax1.set_ylabel('Abwurfwinkel Alpha [rad]')
ax1.set_zlabel('Wurfweite W [m]')
ax1.set_title('Wurfweite als Funktion von v0 und Alpha')
plt.show()

# 2. Plot: Surface mit Colormap
fig2 = plt.figure(2, figsize=(10, 7))
ax2 = fig2.add_subplot(111, projection='3d')
surf = ax2.plot_surface(V0, Alpha, W_values, cmap=cm.viridis, linewidth=0, antialiased=False)
fig2.colorbar(surf, shrink=0.5, aspect=5)
ax2.set_xlabel('Anfangsgeschwindigkeit v0 [m/s]')
ax2.set_ylabel('Abwurfwinkel Alpha [rad]')
ax2.set_zlabel('Wurfweite W [m]')
ax2.set_title('Wurfweite als Funktion von v0 und Alpha')
plt.show()

# 3. Plot: 2D mit Höhenlinien
fig3 = plt.figure(3, figsize=(10, 7))
cont = plt.contour(V0, Alpha, W_values, levels=15, cmap=cm.viridis)
fig3.colorbar(cont, shrink=0.5, aspect=5)
plt.xlabel('Anfangsgeschwindigkeit v0 [m/s]')
plt.ylabel('Abwurfwinkel Alpha [rad]')
plt.title('Höhenlinien der Wurfweite')
plt.show()


# Aufgabe 1 b)
# Zustandsgleichung eines idealen Gases: pV = RT

# Gaskonstante
R = 8.31  # J/(mol*K)

# 1. Funktion: p = p(V, T) = RT/V
# Definitionsbereiche: V in [0.001, 0.2], T in [0, 1e4]
def p_func(V, T):
    return R * T / V

V_range = np.linspace(0.01, 0.2, 100)
T_range1 = np.linspace(0, 1e4, 100)
[V_grid, T_grid1] = np.meshgrid(V_range, T_range1)
p_values = p_func(V_grid, T_grid1)

# Wireframe für p(V, T)
fig4 = plt.figure(4, figsize=(10, 7))
ax4 = fig4.add_subplot(111, projection='3d')
ax4.plot_wireframe(V_grid, T_grid1, p_values, rstride=5, cstride=5)
ax4.set_xlabel('Volumen V [m³]')
ax4.set_ylabel('Temperatur T [K]')
ax4.set_zlabel('Druck p [N/m²]')
ax4.set_title('Druck als Funktion von V und T - Wireframe')
plt.show()

# Surface für p(V, T)
fig5 = plt.figure(5, figsize=(10, 7))
ax5 = fig5.add_subplot(111, projection='3d')
surf = ax5.plot_surface(V_grid, T_grid1, p_values, cmap=cm.plasma, linewidth=0, antialiased=False)
fig5.colorbar(surf, shrink=0.5, aspect=5)
ax5.set_xlabel('Volumen V [m³]')
ax5.set_ylabel('Temperatur T [K]')
ax5.set_zlabel('Druck p [N/m²]')
ax5.set_title('Druck als Funktion von V und T - Surface')
plt.show()

# Höhenlinien für p(V, T)
fig6 = plt.figure(6, figsize=(10, 7))
cont = plt.contour(V_grid, T_grid1, p_values, levels=15, cmap=cm.plasma)
fig6.colorbar(cont, shrink=0.5, aspect=5)
plt.xlabel('Volumen V [m³]')
plt.ylabel('Temperatur T [K]')
plt.title('Höhenlinien des Drucks p(V, T)')
plt.show()


# 2. Funktion: V = V(p, T) = RT/p
# Definitionsbereiche: p in [1e4, 1e5], T in [0, 1e4]
def V_func(p, T):
    return R * T / p

p_range = np.linspace(1e4, 1e5, 100)
T_range2 = np.linspace(0, 1e4, 100)
[p_grid, T_grid2] = np.meshgrid(p_range, T_range2)
V_values = V_func(p_grid, T_grid2)

# Wireframe für V(p, T)
fig7 = plt.figure(7, figsize=(10, 7))
ax7 = fig7.add_subplot(111, projection='3d')
ax7.plot_wireframe(p_grid, T_grid2, V_values, rstride=5, cstride=5)
ax7.set_xlabel('Druck p [N/m²]')
ax7.set_ylabel('Temperatur T [K]')
ax7.set_zlabel('Volumen V [m³]')
ax7.set_title('Volumen als Funktion von p und T - Wireframe')
plt.show()

# Surface für V(p, T)
fig8 = plt.figure(8, figsize=(10, 7))
ax8 = fig8.add_subplot(111, projection='3d')
surf = ax8.plot_surface(p_grid, T_grid2, V_values, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig8.colorbar(surf, shrink=0.5, aspect=5)
ax8.set_xlabel('Druck p [N/m²]')
ax8.set_ylabel('Temperatur T [K]')
ax8.set_zlabel('Volumen V [m³]')
ax8.set_title('Volumen als Funktion von p und T - Surface')
plt.show()

# Höhenlinien für V(p, T)
fig9 = plt.figure(9, figsize=(10, 7))
cont = plt.contour(p_grid, T_grid2, V_values, levels=15, cmap=cm.coolwarm)
fig9.colorbar(cont, shrink=0.5, aspect=5)
plt.xlabel('Druck p [N/m²]')
plt.ylabel('Temperatur T [K]')
plt.title('Höhenlinien des Volumens V(p, T)')
plt.show()


# 3. Funktion: T = T(p, V) = pV/R
# Definitionsbereiche: p in [1e4, 1e6], V in [0, 10]
def T_func(p, V):
    return p * V / R

p_range2 = np.linspace(1e4, 1e6, 100)
V_range2 = np.linspace(0, 10, 100)
[p_grid2, V_grid2] = np.meshgrid(p_range2, V_range2)
T_values = T_func(p_grid2, V_grid2)

# Wireframe für T(p, V)
fig10 = plt.figure(10, figsize=(10, 7))
ax10 = fig10.add_subplot(111, projection='3d')
ax10.plot_wireframe(p_grid2, V_grid2, T_values, rstride=5, cstride=5)
ax10.set_xlabel('Druck p [N/m²]')
ax10.set_ylabel('Volumen V [m³]')
ax10.set_zlabel('Temperatur T [K]')
ax10.set_title('Temperatur als Funktion von p und V - Wireframe')
plt.show()

# Surface für T(p, V)
fig11 = plt.figure(11, figsize=(10, 7))
ax11 = fig11.add_subplot(111, projection='3d')
surf = ax11.plot_surface(p_grid2, V_grid2, T_values, cmap=cm.inferno, linewidth=0, antialiased=False)
fig11.colorbar(surf, shrink=0.5, aspect=5)
ax11.set_xlabel('Druck p [N/m²]')
ax11.set_ylabel('Volumen V [m³]')
ax11.set_zlabel('Temperatur T [K]')
ax11.set_title('Temperatur als Funktion von p und V - Surface')
plt.show()

# Höhenlinien für T(p, V)
fig12 = plt.figure(12, figsize=(10, 7))
cont = plt.contour(p_grid2, V_grid2, T_values, levels=15, cmap=cm.inferno)
fig12.colorbar(cont, shrink=0.5, aspect=5)
plt.xlabel('Druck p [N/m²]')
plt.ylabel('Volumen V [m³]')
plt.title('Höhenlinien der Temperatur T(p, V)')
plt.show()
