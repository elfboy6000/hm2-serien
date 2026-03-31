import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    [33.00, 53.00, 3.32, 3.42, 29.00],
    [31.00, 36.00, 3.10, 3.26, 24.00],
    [33.00, 51.00, 3.18, 3.18, 26.00],
    [37.00, 51.00, 3.39, 3.08, 22.00],
    [36.00, 54.00, 3.20, 3.41, 27.00],
    [35.00, 35.00, 3.03, 3.03, 21.00],
    [59.00, 56.00, 4.78, 4.57, 33.00],
    [60.00, 60.00, 4.72, 4.72, 34.00],
    [59.00, 60.00, 4.60, 4.41, 32.00],
    [60.00, 60.00, 4.53, 4.53, 34.00],
    [34.00, 35.00, 2.90, 2.95, 20.00],
    [60.00, 59.00, 4.40, 4.36, 36.00],
    [60.00, 62.00, 4.31, 4.42, 34.00],
    [60.00, 36.00, 4.27, 3.94, 23.00],
    [62.00, 38.00, 4.41, 3.49, 24.00],
    [62.00, 61.00, 4.39, 4.39, 32.00],
    [90.00, 64.00, 7.32, 6.70, 40.00],
    [90.00, 60.00, 7.32, 7.20, 46.00],
    [92.00, 92.00, 7.45, 7.45, 55.00],
    [91.00, 92.00, 7.27, 7.26, 52.00],
    [61.00, 62.00, 3.91, 4.08, 29.00],
    [59.00, 42.00, 3.75, 3.45, 22.00],
    [88.00, 65.00, 6.48, 5.80, 31.00],
    [91.00, 89.00, 6.70, 6.60, 45.00],
    [63.00, 62.00, 4.30, 4.30, 37.00],
    [60.00, 61.00, 4.02, 4.10, 37.00],
    [60.00, 62.00, 4.02, 3.89, 33.00],
    [59.00, 62.00, 3.98, 4.02, 27.00],
    [59.00, 62.00, 4.39, 4.53, 34.00],
    [37.00, 35.00, 2.75, 2.64, 19.00],
    [35.00, 35.00, 2.59, 2.59, 16.00],
    [37.00, 37.00, 2.73, 2.59, 22.00]])

X = data[:, 0:4]   # 32x4 
y = data[:, 4]     # 32x1

ones = np.ones((len(y), 1))   # 32x1 Spalte mit Einsen
A = np.column_stack([X, ones]) # 32x5

ATA = A.T @ A
ATy = A.T @ y
lam = np.linalg.solve(ATA, ATy)

print(f"  λ1 (T_Tank)   = {lam[0]:.6f}")
print(f"  λ2 (T_Benzin) = {lam[1]:.6f}")
print(f"  λ3 (p_Tank)   = {lam[2]:.6f}")
print(f"  λ4 (p_Benzin) = {lam[3]:.6f}")
print(f"  λ5 (Konstante)= {lam[4]:.6f}")
print()
print("  => m_CH = "
      f"{lam[0]:.4f}·T_Tank + "
      f"{lam[1]:.4f}·T_Benzin + "
      f"{lam[2]:.4f}·p_Tank + "
      f"{lam[3]:.4f}·p_Benzin + "
      f"{lam[4]:.4f}")

# Fit & Fehlerfunktional
m_fit = A @ lam                       
residuen = y - m_fit                   
E = np.sum(residuen**2)                

print(f"=== Fehlerfunktional ===")
print(f"  E = {E:.6f}")

# PLOT

versuche = np.arange(1, 33) 

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(versuche, y,     'ko-', markersize=6, label='Messwerte m_CH')
ax1.plot(versuche, m_fit, 'r--o', markersize=5, label='Fit m_CH')
ax1.set_xlabel('Versuch Nr.')
ax1.set_ylabel('m_CH [g]')
ax1.set_title('Kohlenwasserstoff-Dämpfe – Messwerte vs. Fit')
ax1.legend()
ax1.grid(True)

# Plot 2: Residuen
ax2.bar(versuche, residuen, color='steelblue', alpha=0.7, label='Residuen')
ax2.axhline(0, color='k', linewidth=0.8, linestyle='--')
ax2.set_xlabel('Versuch Nr.')
ax2.set_ylabel('Residuum [g]')
ax2.set_title(f'Residuen  (E = {E:.4f})')
ax2.legend()
ax2.grid(True, axis='y')

plt.tight_layout()
plt.show()