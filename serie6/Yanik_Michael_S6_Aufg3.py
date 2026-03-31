import numpy as np
import matplotlib.pyplot as plt


data=np.array([
    [1971, 2250.],
    [1972, 2500.],
    [1974, 5000.],
    [1978, 29000.],
    [1982, 120000.],
    [1985, 275000.],
    [1989, 1180000.],
    [1989, 1180000.],
    [1993, 3100000.],
    [1997, 7500000.],
    [1999, 24000000.],
    [2000, 42000000.],
    [2002, 220000000.],
    [2003, 410000000.],   
    ])

t = data[:, 0] 
N = data[:, 1]

logN = np.log10(N)

A = np.column_stack([np.ones(len(t)), t -1970])
ATA = A.T @ A
ATy = A.T @ logN
theta = np.linalg.solve(ATA, ATy)

print(f"  theta = {theta[0]:.6f}  → N im Jahr 1970 = 10^theta = {10**theta[0]:.1f} Transistoren")
print(f"  theta = {theta[1]:.6f}  → Zunahme log10(N) pro Jahr")

t_dobbel = np.log10(2) / theta[1]

print("=== Verdopplungszeit ===")
print(f"  t_verdoppl = log10(2) / theta[1] = {t_dobbel:.2f} Jahre")
print("  Moore'sches Gesetz sagt: 1.5 bis 2 Jahre")

t_2015 = 2015
logN_2015 = theta[0] + (t_2015 - 1970) * theta[1]
N_2015 = 10**logN_2015

print("=== Extrapolation Jahr 2015 ===")
print(f"Vorhergesagt: {N_2015:.3e} Transistoren")
print("Tatsächlich:  ~4e9 Transistoren (IBM Z13)")

# PLOT

t_fein = np.linspace(1970, 2020, 500)
logN_fit = theta[0] + (t_fein - 1970) * theta[1]

plt.figure(figsize=(10, 6))

plt.semilogy(t, N, 'ko', markersize=7, label='Messdaten')
plt.semilogy(t_fein, 10**logN_fit, 'r-', linewidth=2, label='Fit')

# Extrapolationspunkt 2015 einzeichnen
plt.semilogy(t_2015, N_2015, 'b^', markersize=12, label=f'Extrapolation 2015: {N_2015:.2e}')
plt.semilogy(t_2015, 4e9,   'g^', markersize=10, label='Tatsächlich 2015: ~4e9')

plt.xlabel('Jahr')
plt.ylabel('Anzahl Transistoren')
plt.title("Prozessorentwicklung / Moore's Law")
plt.legend()
plt.grid(True, which='both', alpha=0.4)
plt.tight_layout()
plt.show()