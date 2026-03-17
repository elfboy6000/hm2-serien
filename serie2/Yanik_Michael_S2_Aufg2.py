import sympy as sp
import numpy as np
sp.init_printing()

# var
x1, x2, x3 = sp.symbols('x1 x2 x3')

# Aufgabe 1a: f: R^2 -> R^2
f1a = sp.Matrix([5*x1*x2, x1**2*x2**2 + x1 + 2*x2])
Xa = sp.Matrix([x1, x2])

Dfa = f1a.jacobian(Xa)
print("Allgemeine Jacobi-Matrix Df_a:")
print(Dfa)

# Auswerten mit (1,2)
Dfa_12 = Dfa.subs([(x1,1), (x2,2)])
print(Dfa_12)
print("Numerisch:")
print(Dfa_12.evalf())

# Aufgabe 1b: f: R^3 -> R^3
f1b = sp.Matrix([
    sp.ln(x1**2 + x2**2) + x3**2,
    sp.exp(x2**2 + x3**2) + x1**2,
    1/(x3**2 + x1**2) + x2**2
])
Xb = sp.Matrix([x1, x2, x3])

Dfb = f1b.jacobian(Xb)
print("Allgemeine Jacobi-Matrix Df_b:")
print(Dfb)

# Auswerten mit (1,2,3)
Dfb_123 = Dfb.subs([(x1,1), (x2,2), (x3,3)])
print("Df_b(1,2,3):")
print(Dfb_123)
print("Numerisch:")
print(Dfb_123.evalf())

print("Python 1a: ", Dfa_12.evalf())
print("Python 1b Zeile 1:", Dfb_123[0,:].evalf())