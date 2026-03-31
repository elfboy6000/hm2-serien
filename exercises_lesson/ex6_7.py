import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4], dtype=np.float64)
y = np.array([6,6.8,10,10.5], dtype=np.float64)

def f1(x):
    return x

def f2(x):
    return np.ones(x.shape)

def f(x, lam):
    return lam[0]*f1(x) + lam[1]*f2(x)

# Fehlergleichungssystem aufbauen
A = np.zeros([x.size, 2])
A[:, 0] = f1(x)
A[:, 1] = f2(x)

# Normalgleichungssystem erstellen
C = A.T @ A
b = A.T @ y

print("Kondition des Normalgleichungssystems: ", np.linalg.cond(C, p=np.inf))

#Lösung
lam = np.linalg.solve(C, b)

plt.figure(1)
plt.clf()
plt.plot(x, y, '*')

xf = np.linspace(np.min(x)-0.5, np.max(x)+0.5, 1000)
plt.plot(xf, f(xf, lam))
plt.show()
