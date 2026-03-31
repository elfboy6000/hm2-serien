import numpy as np
import matplotlib.pyplot as plt

# Daten für Temperatur (T) und Dichte (rho)
T = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.float64)
rho = np.array([999.9, 999.7, 998.2, 995.7, 992.2, 988.1,
                983.2, 977.8, 971.8, 965.3, 958.4], dtype=np.float64)

#  Design-Matrix A: f(T) = a·T^2 + b·T + c
A = np.column_stack([np.ones(len(T)), T, T**2])

# Aufgabe a)
# Variante 1: direkt über Normalgleichungssystem: A^T A λ = A^T y
ATA = A.T @ A          
ATy = A.T @ rho        
lam_direkt = np.linalg.solve(ATA, ATy)

print("=== Ohne QR ===")
print(f"  λ1 (=c) = {lam_direkt[0]:.6f}")
print(f"  λ2 (=b) = {lam_direkt[1]:.6f}")
print(f"  λ3 (=a) = {lam_direkt[2]:.6f}")
print(f"  => f(T) = {lam_direkt[2]:.6f}·T^2 + {lam_direkt[1]:.6f}·T + {lam_direkt[0]:.6f}")

# Variante 2: QR-Zerlegung  
# A = Q·R  =>  Rλ = Q^T y  =>  λ = R^-1 (Q^T y)
Q, R = np.linalg.qr(A)          
QTy = Q.T @ rho                  
lam_qr = np.linalg.solve(R, QTy) 

print("=== Mit QR ===")
print(f"  λ1 (=c) = {lam_qr[0]:.6f}")
print(f"  λ2 (=b) = {lam_qr[1]:.6f}")
print(f"  λ3 (=a) = {lam_qr[2]:.6f}")
print(f"  => f(T) = {lam_qr[2]:.6f}·T^2 + {lam_qr[1]:.6f}·T + {lam_qr[0]:.6f}")

# Aufgabe b)

kond_ATA = np.linalg.cond(ATA)
kond_R   = np.linalg.cond(R)

print("=== Konditionszahlen ===")
print(f"  cond(A^T A) = {kond_ATA:.2f}")
print(f"  cond(R)     = {kond_R:.2f}")
print(f"  => A^T A ist ca. {kond_ATA/kond_R:.1f}x schlechter konditioniert als R wegen des Quadrierens.")

# Aufgabe c)
# polyfit gibt Koeffizienten in absteigender Reihenfolge zurück: [a, b, c]
koeff_polyfit = np.polyfit(T, rho, 2)

print("=== numpy.polyfit ===")
print(f"  a = {koeff_polyfit[0]:.6f}")
print(f"  b = {koeff_polyfit[1]:.6f}")
print(f"  c = {koeff_polyfit[2]:.6f}")

# Aufgabe d)

def f_eval(lam, t):
    # lam = [λ1, λ2, λ3] = [c, b, a]
    return lam[0] + lam[1]*t + lam[2]*t**2

rho_fit_direkt  = f_eval(lam_direkt, T)
rho_fit_qr      = f_eval(lam_qr, T)
rho_fit_polyfit = f_eval(koeff_polyfit[::-1], T)

E_direkt  = np.sum((rho - rho_fit_direkt)**2)
E_qr      = np.sum((rho - rho_fit_qr)**2)
E_polyfit = np.sum((rho - rho_fit_polyfit)**2)

print("=== Fehlerfunktionale ===")
print(f"  E (ohne QR)   = {E_direkt:.10f}")
print(f"  E (mit QR)    = {E_qr:.10f}")
print(f"  E (polyfit)   = {E_polyfit:.10f}")
print(f"  => Alle drei Methoden liefern das gleiche Fehlerfunktional, da sie die gleiche Lösung für λ liefern (innerhalb von Rundungsfehlern).")

# PLOT

T_fein = np.linspace(0, 100, 500) 

plt.figure(figsize=(9, 5))


plt.plot(T, rho, 'ko', markersize=6, label='Messdaten')

# Ohne QR
y_direkt = f_eval(lam_direkt, T_fein)
plt.plot(T_fein, y_direkt, 'b-', linewidth=2, label='Fit (ohne QR)')

# Mit QR
y_qr = f_eval(lam_qr, T_fein)
plt.plot(T_fein, y_qr, 'r--', linewidth=2, label='Fit (mit QR)')

# polyfit
y_polyfit = np.polyval(koeff_polyfit, T_fein)
plt.plot(T_fein, y_polyfit, 'g:', linewidth=2.5, label='numpy.polyfit')

plt.xlabel('Temperatur T [°C]')
plt.ylabel('Dichte ρ [kg/m³]')
plt.title('Wasserdichte – Lineare Ausgleichsrechnung mit und ohne QR')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()